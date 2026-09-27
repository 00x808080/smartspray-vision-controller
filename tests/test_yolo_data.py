"""Offline synthetic export, coordinate, and annotation-retention contracts."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.audit_phenobench import audit_dataset, json_text, sha256_file
from tools.yolo_data import (detector_class, export_dataset, prediction_record,
                             read_jsonl, xywh_to_xyxy, xyxy_to_xywh)


class CoordinateTests(unittest.TestCase):
    def test_foreground_mapping(self):
        self.assertEqual([detector_class(1), detector_class(2)], [0, 1])
        for value in (0, 3, -1, 1.0, True, "1"):
            with self.assertRaises(ValueError):
                detector_class(value)

    def test_one_pixel_edges_and_full_image_round_trip(self):
        for box in ([0, 0, 1, 1], [1023, 767, 1024, 768], [0, 0, 1024, 768], [31, 52, 91, 92]):
            normalized = xyxy_to_xywh(box, [1024, 768])
            self.assertGreater(min(normalized[2:]), 0)
            np.testing.assert_allclose(xywh_to_xyxy(normalized, [1024, 768]), box, atol=1e-10)

    def test_invalid_extents_and_nonfinite_values(self):
        for box in ([0, 0, 0, 1], [0, 0, 1, -1], [-1, 0, 1, 1], [0, 0, 11, 1], [0, 0, float("nan"), 1]):
            with self.assertRaises(ValueError):
                xyxy_to_xywh(box, [10, 10])
        with self.assertRaises(ValueError):
            xywh_to_xyxy([0.5, 0.5, -1, 1], [10, 10])
        with self.assertRaises(ValueError):
            xyxy_to_xywh([0, 0, 1, 1], [0, 10])

    def test_prediction_serialization_keeps_original_dimensions(self):
        record = prediction_record("image", [1920, 1080], [0., 1.], [.9, .4],
                                   [[100, 20, 300, 200], [1919, 1079, 1920, 1080]])
        restored = json.loads(json_text(record))
        self.assertEqual(restored["width"], 1920)
        self.assertEqual(restored["detections"][1]["class_name"], "weed")
        self.assertEqual(restored["detections"][0]["xyxy"], [100., 20., 300., 200.])

    def test_empty_predictions(self):
        self.assertEqual(prediction_record("empty", [16, 16], [], [], [])["detections"], [])

    def test_invalid_predictions_are_rejected(self):
        for cls, score, box in ((2, .5, [0, 0, 1, 1]), (0, float("nan"), [0, 0, 1, 1]), (1, .4, [0, 0, 21, 1])):
            with self.assertRaises(ValueError):
                prediction_record("bad", [20, 10], [cls], [score], [box])
        with self.assertRaises(ValueError):
            prediction_record("bad", [20, 10], [], [.5], [])


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root, self.audit = self.base/"source", self.base/"audit"
        kinds = ("images", "semantics", "plant_instances", "plant_visibility", "leaf_instances", "leaf_visibility")
        for split in ("train", "val"):
            for kind in kinds:
                (self.root/split/kind).mkdir(parents=True)
            for number in range(2):
                name = f"05-15_0000{number}_{split}.png"
                rgb = np.full((64, 96, 3), 60+number, dtype=np.uint8)
                Image.fromarray(rgb).save(self.root/split/"images"/name)
                arrays = {kind: np.zeros((64, 96), dtype=np.uint16) for kind in kinds[1:]}
                if number == 0:
                    for (y, x), label, raw_id, visible in (((0, 0), 3, 1001, 127), ((63, 95), 2, 511, 255)):
                        for kind, value in (("semantics", label), ("plant_instances", raw_id),
                                            ("plant_visibility", visible), ("leaf_instances", raw_id), ("leaf_visibility", visible)):
                            arrays[kind][y, x] = value
                for kind, array in arrays.items():
                    Image.fromarray(array).save(self.root/split/kind/name)
        audit_dataset(self.root, self.audit)

    def export(self, name="derived"):
        output = self.base/name
        return output, export_dataset(self.root, self.audit, output)

    def test_deterministic_manifests_labels_splits_and_sources(self):
        before = {str(p): sha256_file(p) for p in self.root.rglob("*.png")}
        a, report = self.export("a")
        b, second = self.export("b")
        self.assertEqual(report, second)
        for name in ("images.jsonl", "provenance.jsonl", "export.json", "selection.json", "smoke-train.txt", "smoke-val.txt"):
            self.assertEqual((a/name).read_bytes(), (b/name).read_bytes())
        for split in ("train", "val"):
            source_names = {p.name for p in (self.root/split/"images").iterdir()}
            self.assertEqual({p.name for p in (a/"images"/split).iterdir()}, source_names)
            self.assertEqual(report["counts"][split]["objects"], 2)
            self.assertEqual(report["counts"][split]["partial"], 1)
        self.assertEqual(before, {str(p): sha256_file(p) for p in self.root.rglob("*.png")})

    def test_empty_one_pixel_partial_labels_and_provenance(self):
        out, _ = self.export()
        rows = read_jsonl(out/"provenance.jsonl")
        self.assertTrue(rows[0]["partial_semantic"])
        self.assertEqual(rows[0]["raw_instance_id"], 1001)
        label = out/"labels/train/05-15_00000_train.txt"
        for line in label.read_text().splitlines():
            parts = list(map(float, line.split()))
            self.assertIn(parts[0], (0, 1))
            box = xywh_to_xyxy(parts[1:], [96, 64])
            np.testing.assert_allclose([box[2]-box[0], box[3]-box[1]], [1, 1])
        self.assertEqual((out/"labels/train/05-15_00001_train.txt").read_text(), "")

    def test_invalid_annotation_rejected(self):
        path = self.audit/"objects.jsonl"
        rows = read_jsonl(path)
        rows[0]["annotation_valid"] = False
        path.write_text("".join(json_text(r) for r in rows))
        with self.assertRaisesRegex(ValueError, "invalid annotation"):
            self.export()

    def test_changed_membership_and_rgb_are_rejected(self):
        (self.root/"train/images/unexpected.png").write_bytes(b"new")
        with self.assertRaisesRegex(ValueError, "membership"):
            self.export()
        (self.root/"train/images/unexpected.png").unlink()
        image = next((self.root/"train/images").iterdir())
        image.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed since audit"):
            self.export()

    def test_duplicate_float32_label_rejected_instead_of_lost(self):
        path = self.audit/"objects.jsonl"
        rows = read_jsonl(path)
        duplicate = {**rows[0], "raw_instance_id": 1002}
        rows.append(duplicate)
        path.write_text("".join(json_text(r) for r in rows))
        images = read_jsonl(self.audit/"images.jsonl")
        images[0]["object_count"] += 1
        (self.audit/"images.jsonl").write_text("".join(json_text(r) for r in images))
        with self.assertRaisesRegex(ValueError, "deduplicate"):
            self.export()

    def test_native_validator_cpu_reports_actual_precision_without_network(self):
        from tools.ml_runtime import PreservingValidator
        from ultralytics import YOLO
        out, _ = self.export()
        with patch("requests.sessions.Session.request", side_effect=AssertionError("network forbidden")):
            model = YOLO("yolo11n.yaml")
            model.val(validator=PreservingValidator, data=str(out/"dataset.yaml"), imgsz=64,
                      device="cpu", batch=2, workers=0, quantize=None, conf=.001, iou=.7,
                      max_det=300, agnostic_nms=False, plots=False, verbose=False,
                      project=str(self.base), name="validation")
        report = json.loads((self.base/"validation/effective-evaluation.json").read_text())
        self.assertEqual(report["images_seen"], 2)
        self.assertEqual(report["precision"], "FP32")
        self.assertEqual(report["transform_chain"], ["LetterBox", "Format"])

    def test_source_or_existing_output_rejected(self):
        with self.assertRaises(ValueError):
            export_dataset(self.root, self.audit, self.root/"derived")
        out, _ = self.export()
        with self.assertRaises(ValueError):
            export_dataset(self.root, self.audit, out)


class TransformTests(unittest.TestCase):
    def test_real_library_transforms_keep_tiny_partial_boxes_and_classes(self):
        from types import SimpleNamespace
        from tools.ml_runtime import preserving_transforms
        from ultralytics.utils.instance import Instances
        for augment in (True, False):
            for count in (0, 3):
                hyp = SimpleNamespace(hsv_h=.01, hsv_s=.2, hsv_v=.2, flipud=1.0, fliplr=1.0)
                boxes = np.array([[0, 0, 1, 1], [63, 63, 64, 64], [0, 15, 1, 64]], dtype=np.float32)[:count]
                classes = np.array([[0], [1], [1]], dtype=np.float32)[:count]
                labels = {"img": np.zeros((64, 64, 3), dtype=np.uint8), "cls": classes,
                          "instances": Instances(boxes.copy(), bbox_format="xyxy", normalized=False)}
                result = preserving_transforms(64, augment, hyp)(labels)
                self.assertEqual(len(result["bboxes"]), count)
                np.testing.assert_array_equal(result["cls"], classes)
                if count:
                    restored = np.array([xywh_to_xyxy(box.tolist(), [64, 64]) for box in result["bboxes"]])
                    expected = boxes.copy()
                    if augment:
                        expected[:, [0, 2]] = 64-boxes[:, [2, 0]]
                        expected[:, [1, 3]] = 64-boxes[:, [3, 1]]
                    np.testing.assert_allclose(restored, expected, atol=1e-5)

    def test_native_letterbox_inverse_serializes_original_nonsquare_pixels(self):
        from tools import ml_runtime
        from ultralytics.utils.ops import scale_boxes
        import torch
        ratio = 1024/1920
        expected = [100., 20., 300., 200.]
        model_box = torch.tensor([[100*ratio, 20*ratio+224, 300*ratio, 200*ratio+224]])
        original = scale_boxes((1024, 1024), model_box, (1080, 1920))
        record = prediction_record("non_square", [1920, 1080], [1], [.75], original.tolist())
        np.testing.assert_allclose(record["detections"][0]["xyxy"], expected, atol=1e-4)

    def test_validation_has_no_random_augmentation(self):
        from types import SimpleNamespace
        from tools.ml_runtime import preserving_transforms
        chain = preserving_transforms(1024, False, SimpleNamespace())
        self.assertEqual([type(t).__name__ for t in chain.transforms], ["LetterBox", "Format"])


if __name__ == "__main__":
    unittest.main()

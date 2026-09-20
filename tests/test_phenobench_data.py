"""Synthetic, offline acceptance checks for the PhenoBench data boundary."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.phenobench_data import extract_objects


class ExtractObjectsTests(unittest.TestCase):
    def test_single_pixel_half_open_bbox_and_exact_area(self):
        semantics = np.zeros((5, 7), dtype=np.uint8)
        instances = np.zeros((5, 7), dtype=np.uint16)
        semantics[2, 3], instances[2, 3] = 1, 511
        objects, issues = extract_objects(semantics, instances)
        self.assertEqual(issues, [])
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0], {
            "raw_instance_id": 511,
            "class_id": 1,
            "class_name": "crop",
            "semantic_labels": [1],
            "bbox": [3, 2, 4, 3],
            "mask_area_px": 1,
            "partial_semantic": False,
            "visibility_values": [],
            "visibility_pixel_counts": [],
            "official_ignore_candidate": None,
            "touches_image_edge": False,
            "annotation_valid": True,
        })

    def test_each_image_corner_keeps_original_pixel_extent(self):
        for y, x in [(0, 0), (0, 6), (4, 0), (4, 6)]:
            with self.subTest(y=y, x=x):
                semantics = np.zeros((5, 7), dtype=np.uint8)
                instances = np.zeros((5, 7), dtype=np.uint16)
                semantics[y, x], instances[y, x] = 2, 65535
                objects, issues = extract_objects(semantics, instances)
                self.assertEqual(issues, [])
                self.assertEqual(objects[0]["bbox"], [x, y, x + 1, y + 1])
                self.assertEqual(objects[0]["mask_area_px"], 1)
                self.assertTrue(objects[0]["touches_image_edge"])
                self.assertEqual(objects[0]["class_name"], "weed")
                self.assertEqual(objects[0]["raw_instance_id"], 65535)

    def test_many_ids_preserve_full_uint16_range_in_both_namespaces(self):
        raw_ids = np.array([65535, 256, 1, 32768, 255, 257], dtype=np.uint16)
        instances = np.vstack([raw_ids, raw_ids])
        semantics = np.vstack([np.ones(6), np.full(6, 2)]).astype(np.uint8)
        objects, issues = extract_objects(semantics, instances)
        self.assertEqual(issues, [])
        expected = [(class_id, raw_id) for class_id in (1, 2)
                    for raw_id in sorted(raw_ids.tolist())]
        self.assertEqual([(o["class_id"], o["raw_instance_id"]) for o in objects], expected)
        self.assertEqual(sum(o["mask_area_px"] for o in objects), 12)
        self.assertEqual(len(objects), 12)

    def test_disconnected_regions_of_same_instance_remain_one_object(self):
        semantics = np.zeros((7, 8), dtype=np.uint8)
        instances = np.zeros((7, 8), dtype=np.uint16)
        semantics[1, 2:4] = 1
        semantics[5, 6] = 1
        instances[semantics == 1] = 1000
        objects, issues = extract_objects(semantics, instances)
        self.assertEqual(issues, [])
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0]["bbox"], [2, 1, 7, 6])
        self.assertEqual(objects[0]["mask_area_px"], 3)

    def test_background_only_has_no_objects(self):
        objects, issues = extract_objects(np.zeros((2, 3), dtype=np.uint8),
                                         np.zeros((2, 3), dtype=np.uint16))
        self.assertEqual((objects, issues), ([], []))

    def test_regular_and_partial_semantics_preserve_visibility_metadata(self):
        semantics = np.array([[1, 2, 3, 4]], dtype=np.uint8)
        instances = np.array([[10, 20, 30, 40]], dtype=np.uint16)
        visibility = np.array([[255, 128, 127, 0]], dtype=np.uint8)
        objects, issues = extract_objects(semantics, instances, visibility)
        self.assertEqual(issues, [])
        by_id = {obj["raw_instance_id"]: obj for obj in objects}
        for raw_id, label, visible, partial, ignore in [
                (10, 1, 255, False, False), (20, 2, 128, False, False),
                (30, 3, 127, True, True), (40, 4, 0, True, True)]:
            with self.subTest(raw_id=raw_id):
                self.assertEqual(by_id[raw_id]["semantic_labels"], [label])
                self.assertEqual(by_id[raw_id]["visibility_values"], [visible])
                self.assertEqual(by_id[raw_id]["visibility_pixel_counts"], [1])
                self.assertIs(by_id[raw_id]["partial_semantic"], partial)
                self.assertIs(by_id[raw_id]["official_ignore_candidate"], ignore)
                self.assertTrue(by_id[raw_id]["annotation_valid"])

    def test_partial_objects_without_visibility_remain_present(self):
        objects, issues = extract_objects(np.array([[3, 4]], dtype=np.uint8),
                                         np.array([[3, 4]], dtype=np.uint16))
        self.assertEqual(issues, [])
        self.assertEqual([o["class_name"] for o in objects], ["crop", "weed"])
        self.assertTrue(all(o["partial_semantic"] for o in objects))
        self.assertTrue(all(o["official_ignore_candidate"] is None for o in objects))

    def test_same_raw_id_in_crop_and_weed_has_separate_objects(self):
        semantics = np.array([[1, 1, 0, 2]], dtype=np.uint8)
        instances = np.array([[4096, 4096, 0, 4096]], dtype=np.uint16)
        objects, issues = extract_objects(semantics, instances)
        self.assertEqual(issues, [])
        self.assertEqual([(o["class_name"], o["bbox"], o["mask_area_px"]) for o in objects],
                         [("crop", [0, 0, 2, 1], 2), ("weed", [3, 0, 4, 1], 1)])
        self.assertTrue(all(o["raw_instance_id"] == 4096 for o in objects))

    def test_mixed_regular_partial_labels_are_flagged_without_majority_vote(self):
        semantics = np.array([[1, 1, 1, 3]], dtype=np.uint8)
        instances = np.full((1, 4), 700, dtype=np.uint16)
        objects, issues = extract_objects(semantics, instances)
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0]["semantic_labels"], [1, 3])
        self.assertEqual(objects[0]["mask_area_px"], 4)
        self.assertTrue(objects[0]["partial_semantic"])
        self.assertFalse(objects[0]["annotation_valid"])
        self.assertEqual(issues, [{"code": "mixed_regular_partial_labels",
                                   "class_id": 1, "raw_instance_id": 700}])

    def test_visibility_straddling_threshold_is_unknown_and_flagged(self):
        semantics = np.full((1, 4), 1, dtype=np.uint8)
        instances = np.full((1, 4), 700, dtype=np.uint16)
        visibility = np.array([[255, 255, 255, 0]], dtype=np.uint8)
        objects, issues = extract_objects(semantics, instances, visibility)
        obj = objects[0]
        self.assertEqual(obj["visibility_values"], [0, 255])
        self.assertEqual(obj["visibility_pixel_counts"], [1, 3])
        self.assertIsNone(obj["official_ignore_candidate"])
        self.assertFalse(obj["annotation_valid"])
        self.assertEqual(issues, [{"code": "nonuniform_instance_visibility",
                                   "class_id": 1, "raw_instance_id": 700}])

    def test_nonuniform_visibility_is_flagged_even_on_same_side_of_threshold(self):
        semantics = np.array([[3, 3]], dtype=np.uint8)
        instances = np.array([[5, 5]], dtype=np.uint16)
        objects, issues = extract_objects(semantics, instances,
                                         np.array([[0, 127]], dtype=np.uint8))
        self.assertIs(objects[0]["official_ignore_candidate"], True)
        self.assertFalse(objects[0]["annotation_valid"])
        self.assertEqual([i["code"] for i in issues], ["nonuniform_instance_visibility"])

    def test_partial_visibility_disagreement_is_flagged_for_both_directions(self):
        for label, visibility in [(1, 0), (2, 127), (3, 128), (4, 255)]:
            with self.subTest(label=label, visibility=visibility):
                objects, issues = extract_objects(
                    np.array([[label]], dtype=np.uint8),
                    np.array([[5]], dtype=np.uint16),
                    np.array([[visibility]], dtype=np.uint8))
                self.assertFalse(objects[0]["annotation_valid"])
                self.assertEqual([i["code"] for i in issues],
                                 ["partial_label_visibility_disagreement"])

    def test_foreground_without_id_and_background_with_id_are_counted(self):
        semantics = np.array([[1, 2, 3, 4, 0, 0]], dtype=np.uint8)
        instances = np.array([[0, 0, 0, 0, 8, 0]], dtype=np.uint16)
        objects, issues = extract_objects(semantics, instances)
        self.assertEqual(objects, [])
        self.assertEqual(issues, [{"code": "foreground_without_instance", "pixels": 4},
                                  {"code": "instance_on_background", "pixels": 1}])

    def test_instance_id_also_on_background_invalidates_its_foreground_object(self):
        for label, class_id in [(1, 1), (2, 2), (3, 1), (4, 2)]:
            with self.subTest(label=label):
                semantics = np.array([[0, label]], dtype=np.uint8)
                instances = np.array([[7, 7]], dtype=np.uint16)
                objects, issues = extract_objects(semantics, instances)
                self.assertEqual(len(objects), 1)
                self.assertEqual(objects[0]["raw_instance_id"], 7)
                self.assertEqual(objects[0]["bbox"], [1, 0, 2, 1])
                self.assertFalse(objects[0]["annotation_valid"])
                self.assertIn({"code": "instance_on_background", "pixels": 1}, issues)
                self.assertIn({"code": "instance_id_also_on_background",
                               "class_id": class_id, "raw_instance_id": 7}, issues)

    def test_invalid_masks_raise_value_error(self):
        base = np.zeros((2, 3), dtype=np.uint8)
        invalids = [np.zeros((2, 3, 1), dtype=np.uint8),
                    np.zeros((0, 3), dtype=np.uint8),
                    np.zeros((3,), dtype=np.uint8),
                    np.zeros((2, 3), dtype=np.float32),
                    np.zeros((2, 3), dtype=bool),
                    np.zeros((2, 3), dtype=object),
                    np.full((2, 3), -1, dtype=np.int32)]
        for index in range(3):
            for invalid in invalids:
                with self.subTest(index=index, shape=invalid.shape, dtype=invalid.dtype):
                    args = [base.copy(), base.copy(), base.copy()]
                    args[index] = invalid
                    with self.assertRaises(ValueError):
                        extract_objects(*args)

    def test_semantics_instance_and_visibility_upper_limits_are_checked(self):
        for index, too_large in [(0, 5), (1, 65536), (2, 256)]:
            with self.subTest(index=index):
                args = [np.zeros((2, 3), dtype=np.int32) for _ in range(3)]
                args[index][0, 0] = too_large
                with self.assertRaises(ValueError):
                    extract_objects(*args)

    def test_dimensions_must_agree_for_both_annotation_inputs(self):
        for index in (1, 2):
            with self.subTest(index=index):
                args = [np.zeros((2, 3), dtype=np.uint8) for _ in range(3)]
                args[index] = np.zeros((3, 2), dtype=np.uint8)
                with self.assertRaises(ValueError):
                    extract_objects(*args)

    def test_repeated_conversion_is_deterministic_and_does_not_mutate_masks(self):
        semantics = np.array([[2, 0, 1, 3], [4, 2, 1, 0]], dtype=np.uint8)
        instances = np.array([[65535, 0, 500, 500], [1, 65535, 500, 0]], dtype=np.uint16)
        visibility = np.array([[255, 0, 255, 0], [0, 255, 255, 0]], dtype=np.uint8)
        originals = [a.copy() for a in (semantics, instances, visibility)]
        for a in (semantics, instances, visibility):
            a.setflags(write=False)
        first = extract_objects(semantics, instances, visibility)
        second = extract_objects(semantics, instances, visibility)
        self.assertEqual(first, second)
        for original, actual in zip(originals, (semantics, instances, visibility)):
            np.testing.assert_array_equal(actual, original)


class PngLoadingTests(unittest.TestCase):
    def test_uint16_png_keeps_large_instance_values(self):
        from tools.audit_phenobench import load_png
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "instances.png"
            expected = np.array([[0, 1, 255, 256, 32768, 65535]], dtype=np.uint16)
            Image.fromarray(expected).save(path)
            actual, info = load_png(path)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(info["size"], [6, 1])
            self.assertEqual(info["bit_depth"], 16)
            self.assertEqual(info["color_type"], 0)
            self.assertEqual(int(actual.max()), 65535)
            self.assertEqual(actual.dtype.kind, "u")

    def test_non_png_input_is_rejected(self):
        from tools.audit_phenobench import load_png
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fake.png"
            path.write_bytes(b"not an image")
            with self.assertRaises(ValueError):
                load_png(path)


class AuditDatasetTests(unittest.TestCase):
    KINDS = ("images", "semantics", "plant_instances", "plant_visibility",
             "leaf_instances", "leaf_visibility")
    NAME = "05-15_00000_frame.png"

    def setUp(self):
        from tools.audit_phenobench import audit_dataset
        self.audit = audit_dataset
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "dataset"
        for split in ("train", "val"):
            for kind in self.KINDS:
                (self.root / split / kind).mkdir(parents=True)
        self.write_sample("train", self.NAME, 1, 65535, [(1, 2), (1, 3)], 255, 31)
        self.write_sample("val", self.NAME, 4, 300, [(0, 0)], 127, 173)
        (self.root / "test").mkdir()
        (self.root / "test" / "hidden.png").write_bytes(b"unreadable hidden-test sentinel")

    def write_sample(self, split, name, label, raw_id, pixels, visibility, rgb):
        masks = {kind: np.zeros((6, 8), dtype=np.uint16) for kind in self.KINDS[1:]}
        for y, x in pixels:
            masks["semantics"][y, x] = label
            masks["plant_instances"][y, x] = raw_id
            masks["plant_visibility"][y, x] = visibility
            masks["leaf_instances"][y, x] = 1000
            masks["leaf_visibility"][y, x] = visibility
        image = np.full((6, 8, 3), rgb, dtype=np.uint8)
        Image.fromarray(image).save(self.root / split / "images" / name)
        for kind, array in masks.items():
            Image.fromarray(array).save(self.root / split / kind / name)

    @staticmethod
    def read_rows(path):
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_complete_audit_preserves_split_namespace_and_all_six_modalities(self):
        output = self.base / "audit"
        report = self.audit(self.root, output, previews=0)
        self.assertEqual(report["audited_splits"], ["train", "val"])
        self.assertEqual(report["technical_status"], "checks_passed")
        self.assertFalse(report["ready_for_training"])
        self.assertEqual(report["issues"]["total"], 0)
        self.assertEqual(report["preview_count"], 0)
        self.assertEqual(json.loads((output / "report.json").read_text()), report)
        for split in ("train", "val"):
            self.assertEqual(set(report["splits"][split]["files"]), set(self.KINDS))
            for kind in self.KINDS:
                stats = report["splits"][split]["files"][kind]
                self.assertEqual((stats["files"], stats["readable"], stats["corrupt"]), (1, 1, 0))
        rows = self.read_rows(output / "objects.jsonl")
        self.assertEqual([(r["split"], r["filename"], r["class_name"], r["raw_instance_id"])
                          for r in rows],
                         [("train", self.NAME, "crop", 65535), ("val", self.NAME, "weed", 300)])
        self.assertEqual([r["bbox"] for r in rows], [[2, 1, 4, 2], [0, 0, 1, 1]])
        self.assertEqual([r["mask_area_px"] for r in rows], [2, 1])
        self.assertTrue(all(row["image_annotation_valid"] for row in rows))
        self.assertTrue(rows[1]["partial_semantic"])
        self.assertTrue(rows[1]["official_ignore_candidate"])
        self.assertEqual(self.read_rows(output / "issues.jsonl"), [])
        manifest = self.read_rows(output / "file_manifest.jsonl")
        self.assertEqual(len(manifest), 12)
        self.assertEqual({r["path"] for r in manifest},
                         {f"{split}/{kind}/{self.NAME}" for split in ("train", "val") for kind in self.KINDS})
        self.assertEqual(len(self.read_rows(output / "images.jsonl")), 2)

    def test_repeated_audits_have_identical_artifacts_and_preserve_sources(self):
        before = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        for name in ("first", "second"):
            self.audit(self.root, self.base / name, previews=1)
        first = {p.relative_to(self.base / "first"): p.read_bytes()
                 for p in (self.base / "first").rglob("*") if p.is_file()}
        second = {p.relative_to(self.base / "second"): p.read_bytes()
                  for p in (self.base / "second").rglob("*") if p.is_file()}
        self.assertEqual(first, second)
        after = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(len(json.loads(first[Path("previews.json")])), 1)
        preview = json.loads(first[Path("previews.json")])[0]
        self.assertEqual(preview["split"], "train")

    def test_hidden_test_is_never_enumerated_or_opened(self):
        from unittest.mock import patch
        original_iterdir = Path.iterdir
        original_open = Path.open
        hidden = self.root / "test"

        def checked_iterdir(path):
            if path == hidden or hidden in path.parents:
                raise AssertionError("hidden-test enumeration")
            return original_iterdir(path)

        def checked_open(path, *args, **kwargs):
            if path == hidden or hidden in path.parents:
                raise AssertionError("hidden-test open")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "iterdir", checked_iterdir), patch.object(Path, "open", checked_open):
            report = self.audit(self.root, self.base / "audit", previews=1)
        self.assertEqual(report["issues"]["total"], 0)

    def test_symlinked_train_split_cannot_read_validation_or_hidden_test(self):
        from unittest.mock import patch
        original_iterdir = Path.iterdir
        original_open = Path.open
        train = self.root / "train"
        hidden = self.root / "test"
        train.rename(self.root / "preserved-train")

        def checked_iterdir(path):
            if path == train or train in path.parents or path == hidden or hidden in path.parents:
                raise AssertionError("forbidden split enumeration")
            return original_iterdir(path)

        def checked_open(path, *args, **kwargs):
            if path == train or train in path.parents or path == hidden or hidden in path.parents:
                raise AssertionError("forbidden split open")
            return original_open(path, *args, **kwargs)

        for target in ("val", "test"):
            with self.subTest(target=target):
                train.symlink_to(self.root / target, target_is_directory=True)
                output = self.base / ("symlink-" + target)
                try:
                    with patch.object(Path, "iterdir", checked_iterdir), patch.object(Path, "open", checked_open):
                        report = self.audit(self.root, output, previews=1)
                    self.assertEqual(report["technical_status"], "issues_require_review")
                    self.assertEqual(report["issues"]["by_code"]["symlink_split_directory"], 1)
                    self.assertEqual([r["split"] for r in self.read_rows(output / "objects.jsonl")], ["val"])
                    self.assertEqual(report["preview_count"], 0)
                finally:
                    train.unlink()

    def test_exported_objects_carry_whole_image_annotation_validity(self):
        Image.fromarray(np.zeros((6, 8), dtype=np.uint8)).save(
            self.root / "train" / "leaf_instances" / self.NAME)
        output = self.base / "audit"
        report = self.audit(self.root, output)
        self.assertEqual(report["issues"]["by_code"]["unexpected_mask_bit_depth"], 1)
        rows = self.read_rows(output / "objects.jsonl")
        self.assertEqual([row["split"] for row in rows], ["train", "val"])
        self.assertTrue(rows[0]["annotation_valid"])
        self.assertFalse(rows[0]["image_annotation_valid"])
        self.assertTrue(rows[1]["annotation_valid"])
        self.assertTrue(rows[1]["image_annotation_valid"])

    def test_missing_files_are_reported_for_every_modality(self):
        for kind in self.KINDS:
            with self.subTest(kind=kind):
                target = self.root / "train" / kind / self.NAME
                saved = target.read_bytes()
                target.unlink()
                try:
                    output = self.base / ("missing-" + kind)
                    report = self.audit(self.root, output)
                    issues = self.read_rows(output / "issues.jsonl")
                    self.assertEqual(report["issues"]["by_code"]["missing_file"], 1)
                    self.assertIn({"code": "missing_file", "split": "train", "filename": self.NAME,
                                   "directory": kind}, issues)
                    self.assertEqual(report["technical_status"], "issues_require_review")
                    self.assertEqual([r["split"] for r in self.read_rows(output / "objects.jsonl")], ["val"])
                finally:
                    target.write_bytes(saved)

    def test_corrupt_files_are_reported_for_every_modality(self):
        for kind in self.KINDS:
            with self.subTest(kind=kind):
                target = self.root / "train" / kind / self.NAME
                saved = target.read_bytes()
                target.write_bytes(b"invalid PNG")
                try:
                    output = self.base / ("corrupt-" + kind)
                    report = self.audit(self.root, output)
                    self.assertEqual(report["issues"]["by_code"]["unreadable_png"], 1)
                    self.assertEqual(report["splits"]["train"]["files"][kind]["corrupt"], 1)
                    affected = [r for r in self.read_rows(output / "file_manifest.jsonl")
                                if r["path"] == f"train/{kind}/{self.NAME}"]
                    self.assertEqual(affected[0]["status"], "unreadable")
                    self.assertEqual(report["technical_status"], "issues_require_review")
                finally:
                    target.write_bytes(saved)

    def test_dimension_mismatch_in_every_mask_is_reported(self):
        for kind in self.KINDS[1:]:
            with self.subTest(kind=kind):
                target = self.root / "train" / kind / self.NAME
                saved = target.read_bytes()
                Image.fromarray(np.zeros((5, 8), dtype=np.uint16)).save(target)
                try:
                    output = self.base / ("shape-" + kind)
                    report = self.audit(self.root, output)
                    self.assertEqual(report["issues"]["by_code"]["dimension_mismatch"], 1)
                    affected = [r for r in self.read_rows(output / "issues.jsonl")
                                if r["code"] == "dimension_mismatch"]
                    self.assertEqual(affected[0]["directory"], kind)
                    self.assertEqual([r["split"] for r in self.read_rows(output / "objects.jsonl")], ["val"])
                finally:
                    target.write_bytes(saved)

    def test_uint8_visibility_pngs_are_accepted_without_value_conversion(self):
        for split in ("train", "val"):
            for kind in ("plant_visibility", "leaf_visibility"):
                target = self.root / split / kind / self.NAME
                with Image.open(target) as image:
                    array = np.array(image)
                Image.fromarray(array.astype(np.uint8)).save(target)
        output = self.base / "audit"
        report = self.audit(self.root, output)
        self.assertEqual(report["technical_status"], "checks_passed")
        self.assertEqual(report["issues"]["total"], 0)
        for split in ("train", "val"):
            for kind in ("plant_visibility", "leaf_visibility"):
                self.assertEqual(report["splits"][split]["files"][kind]["bit_depths"], {"8": 1})
        rows = self.read_rows(output / "objects.jsonl")
        self.assertEqual([row["visibility_values"] for row in rows], [[255], [127]])
        self.assertEqual([row["official_ignore_candidate"] for row in rows], [False, True])
        self.assertTrue(all(row["image_annotation_valid"] for row in rows))

    def test_unexpected_mask_depth_and_values_block_technical_pass(self):
        Image.fromarray(np.zeros((6, 8), dtype=np.uint8)).save(
            self.root / "train" / "leaf_instances" / self.NAME)
        Image.fromarray(np.full((6, 8), 5, dtype=np.uint16)).save(
            self.root / "train" / "semantics" / self.NAME)
        report = self.audit(self.root, self.base / "audit")
        self.assertEqual(report["technical_status"], "issues_require_review")
        self.assertEqual(report["issues"]["by_code"]["unexpected_mask_bit_depth"], 1)
        self.assertEqual(report["issues"]["by_code"]["unsupported_mask_values"], 1)
        self.assertEqual(report["issues"]["by_code"]["invalid_annotation"], 1)

    def test_duplicate_rgb_and_numeric_overlap_are_reported_without_merging(self):
        self.write_sample("val", self.NAME, 1, 65535, [(2, 3)], 255, 173)
        source = self.root / "train" / "images" / self.NAME
        (self.root / "val" / "images" / self.NAME).write_bytes(source.read_bytes())
        report = self.audit(self.root, self.base / "audit")
        self.assertEqual(len(report["cross_split_image_duplicate_groups"]), 1)
        group = report["cross_split_image_duplicate_groups"][0]
        self.assertEqual(group["paths"], [f"train/images/{self.NAME}", f"val/images/{self.NAME}"])
        self.assertEqual(report["split_overlap"]["source_group_candidates"], ["05-15/frame"])
        self.assertEqual(report["split_overlap"]["crop_ids"], [65535])
        self.assertEqual(report["split_overlap"]["weed_numeric_ids"], [])
        self.assertEqual(len(self.read_rows(self.base / "audit" / "objects.jsonl")), 2)

    def test_background_only_image_is_present_without_objects(self):
        self.write_sample("train", self.NAME, 0, 0, [], 0, 31)
        output = self.base / "audit"
        report = self.audit(self.root, output)
        self.assertEqual(report["issues"]["total"], 0)
        images = self.read_rows(output / "images.jsonl")
        self.assertEqual(images[0]["split"], "train")
        self.assertEqual(images[0]["object_count"], 0)
        self.assertIsNone(images[0]["min_bbox_area_px"])
        self.assertEqual([r["split"] for r in self.read_rows(output / "objects.jsonl")], ["val"])

    def test_missing_required_validation_split_is_explicitly_blocked(self):
        (self.root / "val").rename(self.root / "not-validation")
        output = self.base / "audit"
        report = self.audit(self.root, output)
        self.assertEqual(report["technical_status"], "issues_require_review")
        self.assertEqual(report["issues"]["by_code"]["missing_or_symlink_directory"], 6)
        self.assertEqual(report["issues"]["by_code"]["empty_split"], 1)
        self.assertEqual([r["split"] for r in self.read_rows(output / "objects.jsonl")], ["train"])

    def test_output_inside_source_and_existing_output_are_rejected(self):
        with self.assertRaises(ValueError):
            self.audit(self.root, self.root / "generated")
        existing = self.base / "existing"
        existing.mkdir()
        sentinel = existing / "preserve.txt"
        sentinel.write_text("preserve existing output")
        with self.assertRaises(FileExistsError):
            self.audit(self.root, existing)
        self.assertEqual(sentinel.read_text(), "preserve existing output")

    def test_cli_success_and_detected_issue_have_distinct_exit_codes(self):
        script = Path(__file__).resolve().parents[1] / "tools" / "audit_phenobench.py"
        command = [sys.executable, str(script), "--dataset-root", str(self.root), "--previews", "0"]
        result = subprocess.run(command + ["--output", str(self.base / "success")],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["technical_status"], "checks_passed")
        (self.root / "train" / "plant_instances" / self.NAME).unlink()
        result = subprocess.run(command + ["--output", str(self.base / "failure")],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(json.loads(result.stdout)["technical_status"], "issues_require_review")

    def test_preview_count_out_of_bounds_is_rejected_without_creating_output(self):
        for count in (-1, 13):
            with self.subTest(count=count):
                output = self.base / f"preview-{count}"
                with self.assertRaises(ValueError):
                    self.audit(self.root, output, previews=count)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

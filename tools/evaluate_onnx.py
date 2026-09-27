#!/usr/bin/env python3
"""Evaluate the frozen FP32 ONNX graph once with the unchanged M2.2 AP protocol."""
import argparse
import importlib.metadata
import json
import math
from pathlib import Path
from unittest.mock import patch

BASELINE_SETTINGS = dict(imgsz=1024, conf=0.001, iou=0.7, max_det=300,
                         agnostic_nms=False, quantize=None, augment=False, device=0,
                         batch=8, workers=4, rect=False, plots=True, verbose=False)
EVALUATION = {**BASELINE_SETTINGS, "device": "cpu", "batch": 1, "workers": 0, "plots": False}
METRIC_NAMES = ("mAP50-95", "AP50", "AP75")
NAMES = {0: "crop", 1: "weed"}
EXPECTED_IMAGES = 772
EXPECTED_OBJECTS = 10408
PROTOCOL = "all-annotation-valid-appearances-including-partial-v1"


def summarize_ap(ap, class_indices):
    """Extract exactly the native M2.2 AP columns; no new matching or AP calculation."""
    import numpy as np
    ap = np.asarray(ap, dtype=np.float64)
    classes = [int(x) for x in class_indices]
    if ap.shape != (2, 10) or sorted(classes) != [0, 1] or not np.isfinite(ap).all():
        raise ValueError("Expected finite native AP for both classes and ten IoU thresholds")
    if np.any(ap < 0) or np.any(ap > 1):
        raise ValueError("Native AP values must be between zero and one")
    rows = {NAMES[cls]: {"class_id": cls, "mAP50-95": float(ap[i].mean()),
                       "AP50": float(ap[i, 0]), "AP75": float(ap[i, 5])}
            for i, cls in enumerate(classes)}
    return {"aggregate": {"mAP50-95": float(ap.mean()), "AP50": float(ap[:, 0].mean()),
                          "AP75": float(ap[:, 5].mean())},
            "classes": rows, "ap_per_iou": ap.tolist(),
            "ap_class_index": classes, "iou_thresholds": [round(.5 + .05 * i, 2) for i in range(10)]}


def metric_deltas(baseline, current, threshold):
    """Preserve signed and absolute differences, and flag every metric above threshold."""
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("Investigation threshold must be finite and nonnegative")
    flagged = []
    def compare(old, new, prefix):
        result = {}
        for name in METRIC_NAMES:
            before, after = float(old[name]), float(new[name])
            if not math.isfinite(before) or not math.isfinite(after):
                raise ValueError("Cannot compare nonfinite metrics")
            signed = after - before
            exceeded = abs(signed) > threshold
            result[name] = {"baseline": before, "onnx": after, "signed_delta": signed,
                            "absolute_delta": abs(signed), "exceeds_investigation_threshold": exceeded}
            if exceeded:
                flagged.append(prefix + "." + name)
        return result
    aggregate = compare(baseline["aggregate"], current["aggregate"], "aggregate")
    classes = {name: compare(baseline["classes"][name], current["classes"][name], name)
               for name in NAMES.values()}
    return {"aggregate": aggregate, "classes": classes,
            "investigation_threshold": threshold, "threshold_comparison": "absolute_delta > threshold",
            "investigation_required": bool(flagged), "metrics_requiring_investigation": flagged,
            "units": "AP fraction in [0,1]; 0.005 equals 0.5 percentage points"}


def validate_baseline(baseline, expected_checkpoint_sha256):
    if baseline["checkpoint_sha256"] != expected_checkpoint_sha256:
        raise ValueError("Original baseline checkpoint hash differs from the frozen checkpoint")
    if baseline["settings"] != BASELINE_SETTINGS or baseline["protocol"] != PROTOCOL:
        raise ValueError("Original baseline metric/data settings changed")
    if baseline["images"] != EXPECTED_IMAGES or baseline["objects"] != EXPECTED_OBJECTS:
        raise ValueError("Original baseline validation counts changed")
    if baseline["inference_precision"] != "FP32":
        raise ValueError("Original baseline precision must be FP32")
    if {row["class_id"]: name for name, row in baseline["classes"].items()} != NAMES:
        raise ValueError("Original baseline class mapping changed")


def session_record(session):
    import onnxruntime as ort
    options = session.get_session_options()
    if (session.get_providers() != ["CPUExecutionProvider"] or
            options.intra_op_num_threads != 1 or options.inter_op_num_threads != 1 or
            options.execution_mode != ort.ExecutionMode.ORT_SEQUENTIAL or
            options.graph_optimization_level != ort.GraphOptimizationLevel.ORT_ENABLE_ALL):
        raise ValueError("Actual ONNX session differs from frozen CPU deployment options")
    return {"providers": session.get_providers(), "provider_options": session.get_provider_options(),
            "intra_op_threads": options.intra_op_num_threads,
            "inter_op_threads": options.inter_op_num_threads,
            "execution_mode": "sequential", "graph_optimization": "all"}


def evaluate(args):
    # Import the deployment environment guard first. ml_runtime retains its existing
    # local font/settings support; neither M2.2 source nor installed packages change.
    try:
        from . import onnx_deploy as ref
    except ImportError:
        import onnx_deploy as ref
    output, derived = args.output.resolve(), args.derived.resolve()
    model_path, baseline_path = args.model.resolve(), args.baseline.resolve()
    if output.exists():
        raise ValueError("Evaluation output must be new; refusing to repeat or overwrite a run")
    if model_path.suffix != ".onnx":
        raise ValueError("Expected the exported .onnx graph")
    baseline_hash = ref.sha256(baseline_path)
    baseline = json.loads(baseline_path.read_text())
    validate_baseline(baseline, ref.CHECKPOINT_SHA256)
    graph_hash = ref.sha256(model_path)
    export_path = model_path.with_suffix(".json")
    exported = json.loads(export_path.read_text())
    if (exported["onnx_sha256"] != graph_hash or
            exported["checkpoint_sha256_before_and_after"] != ref.CHECKPOINT_SHA256 or
            exported["arguments"] != dict(format="onnx", imgsz=1024, batch=1, device="cpu", quantize=None,
                                           nms=False, dynamic=False, simplify=False, opset=18)):
        raise ValueError("Graph SHA256, checkpoint provenance or export arguments differ")
    versions = {name: importlib.metadata.version(name)
                for name in ("torch", "ultralytics", "onnx", "onnxruntime")}
    if versions != exported["versions"] or versions["onnxruntime"] != "1.22.0":
        raise ValueError(f"Evaluation environment differs from verified export: {versions}")
    checked_session = ref.session(model_path)
    checked_runtime = session_record(checked_session)
    del checked_session

    try:
        from .ml_runtime import PreservingValidator
        from .yolo_data import read_jsonl
    except ImportError:
        from ml_runtime import PreservingValidator
        from yolo_data import read_jsonl
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.nn.autobackend import AutoBackend
    from ultralytics.nn.backends.onnx import ONNXBackend
    import ultralytics.models.yolo.detect.val as native_val
    import ultralytics.utils.metrics as native_metrics
    import ultralytics.utils.nms as native_nms

    torch.set_num_threads(1)
    images = [row for row in read_jsonl(derived / "images.jsonl") if row["split"] == "val"]
    if (len(images) != EXPECTED_IMAGES or sum(row["object_count"] for row in images) != EXPECTED_OBJECTS
            or len({row["image_id"] for row in images}) != EXPECTED_IMAGES):
        raise ValueError("Validation image membership/counts differ from the frozen dataset")
    maximum = max(row["object_count"] for row in images)
    if maximum != baseline["max_objects"] or maximum > EVALUATION["max_det"]:
        raise ValueError("Validation object maximum differs or exceeds max_det")
    for row in images:
        label = derived / "labels/val" / (row["image_id"] + ".txt")
        if ref.sha256(label) != row["label_sha256"]:
            raise ValueError(f"Validation label hash changed: {row['image_id']}")
    data_path = derived / "dataset.yaml"
    source_paths = [Path(__file__), Path(__file__).with_name("ml_runtime.py"),
                    Path(__file__).with_name("evaluate_baseline.py"),
                    Path(native_val.__file__), Path(native_metrics.__file__), Path(native_nms.__file__)]
    evidence = {"model_path": str(model_path), "onnx_sha256": graph_hash,
                "checkpoint_sha256": ref.CHECKPOINT_SHA256,
                "export_record_sha256": ref.sha256(export_path), "baseline_metrics_path": str(baseline_path),
                "baseline_metrics_sha256": baseline_hash,
                "dataset_yaml_sha256": ref.sha256(data_path),
                "images_manifest_sha256": ref.sha256(derived / "images.jsonl"),
                "validation_labels_sha256_verified": len(images),
                "source_sha256": {str(path.resolve()): ref.sha256(path) for path in source_paths}}
    output.mkdir(parents=True, exist_ok=False)
    ref.write_json(output / "preflight.json", {**evidence, "settings": EVALUATION,
                   "runtime": checked_runtime, "versions": versions, "images": len(images),
                   "objects": sum(row["object_count"] for row in images)})
    backend_records, observed = [], {}
    def frozen_backend(*positional, **keywords):
        backend = ONNXBackend(*positional, **keywords, session_options=ref.session_options())
        if backend.names != NAMES or backend.fp16 or backend.device.type != "cpu":
            raise ValueError("Native ONNX backend class/precision/device mismatch")
        backend_records.append(session_record(backend.session))
        return backend
    def capture_validation(validator):
        observed.update(images_seen=int(validator.seen),
                        objects_seen=int(validator.metrics.nt_per_class.sum()),
                        objects_per_class={NAMES[i]: int(n) for i, n in enumerate(validator.metrics.nt_per_class)},
                        names=validator.names, effective_args=dict(vars(validator.args)),
                        transform_chain=[type(t).__name__ for t in validator.dataloader.dataset.transforms.transforms],
                        image_ids=sorted(Path(path).stem for path in validator.dataloader.dataset.im_files))
    model = YOLO(str(model_path), task="detect")
    model.add_callback("on_val_end", capture_validation)
    with patch.dict(AutoBackend._BACKEND_MAP, {"onnx": frozen_backend}):
        metrics = model.val(validator=PreservingValidator, data=str(data_path),
                            project=str(output.parent), name=output.name, exist_ok=True, **EVALUATION)
    if len(backend_records) != 1:
        raise RuntimeError("Expected one frozen native ONNX evaluation session")
    if (observed["images_seen"] != EXPECTED_IMAGES or observed["objects_seen"] != EXPECTED_OBJECTS or
            observed["image_ids"] != sorted(row["image_id"] for row in images) or observed["names"] != NAMES or
            observed["transform_chain"] != ["LetterBox", "Format"]):
        raise RuntimeError("Actual native validation changed image/label/class/transform membership")
    for key, expected in EVALUATION.items():
        if observed["effective_args"][key] != expected:
            raise RuntimeError(f"Native validator changed frozen setting {key}")
    if ref.sha256(model_path) != graph_hash or ref.sha256(baseline_path) != baseline_hash:
        raise RuntimeError("ONNX model or baseline metric evidence changed during evaluation")
    summary = {**summarize_ap(metrics.box.all_ap, metrics.box.ap_class_index), **evidence,
               "protocol": PROTOCOL, "evaluator": f"ultralytics {ultralytics.__version__} PreservingValidator",
               "settings": EVALUATION, "effective_settings": observed["effective_args"],
               "inference_precision": "FP32", "images": observed["images_seen"],
               "objects": observed["objects_seen"], "objects_per_class": observed["objects_per_class"],
               "max_objects": maximum, "transform_chain": observed["transform_chain"],
               "native_validation_multi_label": True, "runtime": backend_records[0], "versions": versions,
               "settings_differences_from_m2_2": {
                   "device": {"before": 0, "after": "cpu", "reason": "frozen CPUExecutionProvider deployment"},
                   "batch": {"before": 8, "after": 1, "reason": "frozen ONNX batch dimension is one"},
                   "workers": {"before": 4, "after": 0, "reason": "native CPU data-loading setting; no metric change"},
                   "plots": {"before": True, "after": False, "reason": "no duplicate plots; no metric change"}},
               "metric_semantics": "Unchanged native DetectionValidator AP and multi_label=True NMS at conf=0.001; deployment confidence 0.25 and single-label filtering are not used for AP",
               "P_R_not_reported": baseline["P_R_not_reported"],
               "validation_is_not_independent_unseen_fields": True}
    summary["comparison_to_m2_2"] = metric_deltas(baseline, summary, ref.GATES["map_investigation_threshold"])
    ref.write_json(output / "metrics.json", summary)
    print(json.dumps({"output": str(output), "images": summary["images"], "objects": summary["objects"],
                      "aggregate": summary["aggregate"], "classes": summary["classes"],
                      "comparison_to_m2_2": summary["comparison_to_m2_2"]}, indent=2, allow_nan=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--derived", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()

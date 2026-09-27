#!/usr/bin/env python3
"""Frozen FP32 YOLO11 deployment reference and explicit ONNX export."""
import argparse
import ast
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

# Must precede all Ultralytics imports. Never install packages during inference/export.
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["YOLO_OFFLINE"] = "true"
os.environ["YOLO_CONFIG_DIR"] = str(Path(__file__).resolve().parents[1] / ".local/m3/settings")
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
os.environ["WANDB_DISABLED"] = "true"

import cv2
import numpy as np

NAMES = {0: "crop", 1: "weed"}
SIZE = 1024
CHECKPOINT_SHA256 = "ea861ff9ba54a768dffd62814944d157e50037a67fd268ea65fe0d7773196fd6"
RUNTIME = {"provider": "CPUExecutionProvider", "intra_op_threads": 1,
           "inter_op_threads": 1, "execution_mode": "sequential", "graph_optimization": "all"}
DEPLOYMENT = {"confidence": 0.25, "nms_iou": 0.7, "max_det": 300, "multi_label": False,
              "confidence_comparison": ">", "suppression_comparison": ">",
              "tie_break": "score descending, original candidate index ascending; class tie chooses crop"}
GATES = {"pixels_exact": True, "geometry_exact": True, "input_no_resize_atol": 1e-6,
         "input_resize_atol": 1/255+1e-6, "raw_atol": 1e-4, "raw_rtol": 1e-4,
         "final_box_atol_px": 0.5, "final_score_atol": 1e-4,
         "map_investigation_threshold": 0.005}
GEOMETRY = {"width_m": 2.0, "lookahead_m": 2.0, "nozzle_pitch_m": 0.25,
            "speed_mps": 2.0, "capture_time_us": 1000000, "now_us": 1100000,
            "actuator_delay_us": 50000, "pulse_duration_us": 100000}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")


def session_options():
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return opts


def session(path):
    import onnxruntime as ort
    s = ort.InferenceSession(str(path), sess_options=session_options(), providers=["CPUExecutionProvider"])
    inputs, outputs = s.get_inputs(), s.get_outputs()
    if len(inputs) != 1 or inputs[0].type != "tensor(float)" or inputs[0].shape != [1, 3, SIZE, SIZE]:
        raise ValueError("model input must be one fixed FP32 [1,3,1024,1024] tensor")
    if (len(outputs) != 1 or outputs[0].type != "tensor(float)" or len(outputs[0].shape) != 3
            or outputs[0].shape[:2] != [1, 6] or not isinstance(outputs[0].shape[2], int)
            or outputs[0].shape[2] <= 0):
        raise ValueError("model output must be fixed FP32 [1,6,N>0]")
    meta = s.get_modelmeta().custom_metadata_map
    if ast.literal_eval(meta.get("names", "{}")) != NAMES or meta.get("task") != "detect":
        raise ValueError("model class metadata must be crop=0, weed=1, task=detect")
    if meta.get("end2end", "False") != "False":
        raise ValueError("end-to-end/NMS output is unsupported")
    return s


def decode(path):
    path = Path(path)
    if path.suffix.lower() not in {".png", ".bmp"}:
        raise ValueError("only 8-bit three-channel PNG/BMP images are supported")
    with path.open("rb") as source:
        signature = source.read(8)
    if (path.suffix.lower() == ".png" and signature != b"\x89PNG\r\n\x1a\n") or (path.suffix.lower() == ".bmp" and signature[:2] != b"BM"):
        raise ValueError("image encoding disagrees with supported extension")
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None or im.dtype != np.uint8 or im.ndim != 3 or im.shape[2] != 3:
        raise ValueError("unreadable image or unsupported depth/channels")
    return im


def preprocess(image):
    if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected nonempty uint8 BGR image")
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("empty image")
    r = min(SIZE/h, SIZE/w)  # scaleup=True, auto=False
    rw, rh = round(w*r), round(h*r)  # Python ties-to-even
    if rw <= 0 or rh <= 0:
        raise ValueError("image aspect ratio rounds a resized dimension to zero")
    dw, dh = (SIZE-rw)/2, (SIZE-rh)/2
    left, right, top, bottom = round(dw-.1), round(dw+.1), round(dh-.1), round(dh+.1)
    resized = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR) if (rw, rh) != (w, h) else image
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114))
    tensor = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)/np.float32(255)
    geometry = dict(r=r, left=left, right=right, top=top, bottom=bottom,
                    resized_width=rw, resized_height=rh, original_width=w, original_height=h)
    return tensor, geometry


def valid_raw(raw):
    if raw.dtype != np.float32 or raw.ndim != 3 or raw.shape[:2] != (1, 6) or raw.shape[2] <= 0:
        raise ValueError("expected decoded FP32 [1,6,N>0]")
    if not np.isfinite(raw).all() or np.any(raw[:, 2:4] < 0) or np.any(raw[:, 4:] < 0) or np.any(raw[:, 4:] > 1):
        raise ValueError("invalid/nonfinite decoded boxes or class probabilities")


def iou(a, b):
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    wh = np.maximum(np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2]), np.float32(0))
    inter = wh[0]*wh[1]
    aw, bw = np.maximum(a[2:]-a[:2], 0), np.maximum(b[2:]-b[:2], 0)
    union = aw[0]*aw[1]+bw[0]*bw[1]-inter
    return float(inter/union) if union > 0 else 0.0


def restore(box, g):
    b = np.array(box, dtype=np.float32, copy=True)
    b[[0,2]] = (b[[0,2]]-np.float32(g["left"]))/np.float32(g["r"])
    b[[1,3]] = (b[[1,3]]-np.float32(g["top"]))/np.float32(g["r"])
    b[[0,2]] = np.clip(b[[0,2]], 0, g["original_width"])
    b[[1,3]] = np.clip(b[[1,3]], 0, g["original_height"])
    return b.tolist()


def record(detections, g):
    return {"schema_version": 1, "original_width": g["original_width"], "original_height": g["original_height"],
            "classes": {str(k): v for k,v in NAMES.items()}, "detections": detections}


def postprocess(raw, g):
    valid_raw(raw)
    a = raw[0].T
    classes = a[:,4:].argmax(axis=1)
    scores = a[np.arange(len(a)), classes+4]
    indices = np.flatnonzero(scores > np.float32(DEPLOYMENT["confidence"]))
    indices = sorted(indices, key=lambda i: (-float(scores[i]), int(i)))
    boxes = np.concatenate([a[:,:2]-a[:,2:4]/np.float32(2), a[:,:2]+a[:,2:4]/np.float32(2)], axis=1)
    kept = []
    for i in indices:
        if any(classes[j] == classes[i] and iou(boxes[i], boxes[j]) > np.float32(DEPLOYMENT["nms_iou"]) for j in kept):
            continue
        kept.append(i)
        if len(kept) == DEPLOYMENT["max_det"]:
            break
    return record([{"candidate_index": int(i), "class_id": int(classes[i]), "class_name": NAMES[int(classes[i])],
                    "score": float(scores[i]), "xyxy": restore(boxes[i], g)} for i in kept], g)


def native_postprocess(raw, g):
    import torch
    from ultralytics.utils.nms import non_max_suppression
    valid_raw(raw)
    out, ids = non_max_suppression(torch.from_numpy(raw.copy()), conf_thres=.25, iou_thres=.7,
                                   multi_label=False, agnostic=False, max_det=300, nc=2,
                                   max_time_img=60, return_idxs=True)
    return record([{"candidate_index": int(i), "class_id": int(row[5]), "class_name": NAMES[int(row[5])],
                    "score": float(row[4]), "xyxy": restore(row[:4].numpy(), g)}
                   for row, i in zip(out[0], ids[0])], g)


def load_torch(path):
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(1)
    model = deepcopy(YOLO(str(path)).model).cpu().float().eval()
    if model.names != NAMES or model.model[-1].nc != 2 or model.end2end:
        raise ValueError("checkpoint head/class mismatch")
    # The native predictor and exporter both fuse Conv/BN for inference.
    model.fuse(verbose=False, imgsz=SIZE)
    return model


def infer_torch(model, tensor):
    import torch
    with torch.inference_mode():
        y = model(torch.from_numpy(tensor))
    decoded = y[0] if isinstance(y, (tuple, list)) else y
    raw = decoded.detach().cpu().numpy()
    valid_raw(raw)
    return raw


def export(args):
    import onnx
    import torch
    import ultralytics
    from ultralytics.engine.exporter import Exporter
    if sha256(args.checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("checkpoint SHA256 mismatch; no export")
    expected = {"torch": "2.13.0+cu126", "ultralytics": "8.4.163", "onnx": "1.18.0", "onnxruntime": "1.22.0"}
    actual = {n: importlib.metadata.version(n) for n in expected}
    if actual != expected:
        raise ValueError(f"unverified export environment: {actual}")
    if args.output.suffix != ".onnx" or args.output.exists():
        raise ValueError("output must be a new .onnx path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    model = load_torch(args.checkpoint)
    model.pt_path = str(args.output.with_suffix(".pt"))
    options = dict(format="onnx", imgsz=SIZE, batch=1, device="cpu", quantize=None,
                   nms=False, dynamic=False, simplify=False, opset=18)
    exported = Exporter(overrides=options)(model=model)
    if Path(exported).resolve() != args.output.resolve():
        raise RuntimeError("unexpected exporter output path")
    graph = onnx.load(exported)
    onnx.checker.check_model(graph, full_check=True)
    if any(n.op_type == "NonMaxSuppression" for n in graph.graph.node):
        raise RuntimeError("unexpected embedded NMS")
    if [(op.domain, op.version) for op in graph.opset_import] != [("",18)]:
        raise RuntimeError("unexpected opset")
    runtime = session(exported)
    y = runtime.run(None, {runtime.get_inputs()[0].name: np.zeros((1,3,SIZE,SIZE),np.float32)})[0]
    valid_raw(y)
    if sha256(args.checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("original checkpoint changed")
    write_json(args.output.with_suffix(".json"), {
        "checkpoint_sha256_before_and_after": CHECKPOINT_SHA256, "onnx_sha256": sha256(exported),
        "versions": actual, "arguments": options, "ir_version": graph.ir_version,
        "opsets": [{"domain": op.domain, "version": op.version} for op in graph.opset_import],
        "inputs": [{"name": x.name, "type": x.type, "shape": x.shape} for x in runtime.get_inputs()],
        "outputs": [{"name": x.name, "type": x.type, "shape": x.shape} for x in runtime.get_outputs()],
        "metadata": runtime.get_modelmeta().custom_metadata_map, "runtime": RUNTIME,
        "semantics": "decoded cx,cy,w,h in 1024 input pixels; crop and weed probabilities; no objectness, no extra sigmoid",
        "checker": "PASS", "finite_zero_input": True, "manual_IR_or_opset_changes": False,
        "source_sha256": {str(p): sha256(p) for p in [Path(ultralytics.__file__).parent/"engine/exporter.py",
                           Path(ultralytics.__file__).parent/"utils/export/engine.py"]}})
    print(f"Export/check/load PASS: {exported}, {list(y.shape)}, IR {graph.ir_version}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    export(parser.parse_args())


if __name__ == "__main__":
    main()

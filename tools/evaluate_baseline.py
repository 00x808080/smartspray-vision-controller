#!/usr/bin/env python3
"""Fresh-process native AP evaluation and original-pixel inspection predictions."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

try:
    from .ml_runtime import PreservingValidator
    from .yolo_data import NAMES, POLICY, prediction_record, read_jsonl, sha256_file
except ImportError:
    from ml_runtime import PreservingValidator
    from yolo_data import NAMES, POLICY, prediction_record, read_jsonl, sha256_file

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import ultralytics

EVALUATION = dict(imgsz=1024, conf=0.001, iou=0.7, max_det=300,
                  agnostic_nms=False, quantize=None, augment=False, device=0, batch=8,
                  workers=4, rect=False, plots=True, verbose=False)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(x1-x0, 0)*max(y1-y0, 0)
    return intersection/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection)


def inspection_failures(gt, predictions):
    """IoU>=0.5 qualitative selection only; this is not an AP evaluator."""
    missed_weeds, confusion = [], []
    for obj in gt:
        cls = obj["class_id"]-1
        correct = [p for p in predictions if p["class_id"] == cls and iou(obj["bbox"], p["xyxy"]) >= .5]
        wrong = [p for p in predictions if p["class_id"] != cls and iou(obj["bbox"], p["xyxy"]) >= .5]
        if not correct and cls == 1:
            missed_weeds.append({"raw_instance_id": obj["raw_instance_id"], "bbox": obj["bbox"],
                                 "partial": obj["partial_semantic"]})
        if not correct and wrong:
            confusion.append({"gt_class": NAMES[cls], "predicted_class": wrong[0]["class_name"], "bbox": obj["bbox"]})
    return {"missed_weeds": missed_weeds, "class_confusions": confusion}


def draw_overlay(source, target, record, gt, categories):
    with Image.open(source) as image:
        image = image.convert("RGB")
        width, height = image.size
        canvas = Image.new("RGB", (width*2, height+72), "#111111")
        canvas.paste(image, (0, 72))
        canvas.paste(image, (width, 72))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=17)
    draw.text((8, 4), record["image_id"]+" | "+", ".join(categories), fill="white", font=font)
    draw.text((8, 29), "GROUND TRUTH | green=crop, red=weed | partial retained", fill="white", font=font)
    draw.text((width+8, 29), "PREDICTIONS | blue=crop, magenta=weed | confidence >= 0.25", fill="white", font=font)
    for obj in gt:
        x0, y0, x1, y1 = obj["bbox"]
        color = "lime" if obj["class_id"] == 1 else "#ff5544"
        draw.rectangle((x0, y0+72, x1-1, y1-1+72), outline=color, width=2)
        label = obj["class_name"]+(" partial" if obj["partial_semantic"] else "")
        draw.text((min(x0, width-150), max(72, y0+52)), label, fill=color, font=font, stroke_width=1, stroke_fill="black")
    for obj in record["detections"]:
        x0, y0, x1, y1 = obj["xyxy"]
        color = "#55aaff" if obj["class_id"] == 0 else "#ff55ff"
        draw.rectangle((width+x0, y0+72, width+x1, y1+72), outline=color, width=2)
        draw.text((width+min(x0, width-150), max(72, y0+52)), f"{obj['class_name']} {obj['score']:.2f}",
                  fill=color, font=font, stroke_width=1, stroke_fill="black")
    canvas.save(target)


def evaluate(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing silent CPU fallback")
    output, derived = args.output.resolve(), args.derived.resolve()
    if output.exists():
        raise ValueError("evaluation output must be new")
    model = YOLO(str(args.weights.resolve()))
    if model.names != NAMES:
        raise ValueError(f"checkpoint class mapping differs: {model.names}")
    images = [r for r in read_jsonl(derived/"images.jsonl") if r["split"] == "val"]
    selection = json.loads((derived/"selection.json").read_text())
    if args.smoke:
        images = [r for r in images if r["image_id"] in selection["smoke"]["val"]]
    data = derived/("smoke.yaml" if args.smoke else "dataset.yaml")
    maximum = max(r["object_count"] for r in images)
    if maximum > EVALUATION["max_det"]:
        raise ValueError("frozen max_det is below an actual GT count")
    metrics = model.val(validator=PreservingValidator, data=str(data),
                        project=str(output.parent), name=output.name, exist_ok=False, **EVALUATION)
    ap = metrics.box.all_ap
    class_rows = {}
    for index, cls in enumerate(metrics.box.ap_class_index):
        class_rows[NAMES[int(cls)]] = {"class_id": int(cls), "mAP50-95": float(ap[index].mean()),
                                      "AP50": float(ap[index, 0]), "AP75": float(ap[index, 5])}
    if set(class_rows) != set(NAMES.values()):
        raise RuntimeError("evaluator did not report both annotated classes")
    summary = {"protocol": POLICY, "evaluator": f"ultralytics {ultralytics.__version__} DetectionValidator",
               "settings": EVALUATION, "inference_precision": "FP32", "images": len(images),
               "objects": sum(r["object_count"] for r in images), "max_objects": maximum,
               "checkpoint_sha256": sha256_file(args.weights),
               "aggregate": {"mAP50-95": float(ap.mean()), "AP50": float(ap[:, 0].mean()), "AP75": float(ap[:, 5].mean())},
               "classes": class_rows, "ap_per_iou": ap.tolist(),
               "iou_thresholds": [round(.5+.05*i, 2) for i in range(10)],
               "P_R_not_reported": "native summaries use the confidence maximizing mean smoothed F1 at IoU 0.5, not fixed confidence",
               "validation_is_not_independent_unseen_fields": True}
    write_json(output/"metrics.json", summary)
    gt = defaultdict(list)
    for row in read_jsonl(derived/"provenance.jsonl"):
        if row["split"] == "val":
            gt[row["image_id"]].append(row)
    paths = [str(derived/"images/val"/r["filename"]) for r in images]
    prediction_settings = {k: v for k, v in EVALUATION.items() if k not in ("plots", "workers")}
    prediction_settings.update(conf=.25, save=False, stream=True)
    source_list = output/"prediction-images.txt"
    source_list.write_text("\n".join(paths)+"\n")
    records, failures = [], {}
    for result in model.predict(source=str(source_list), **prediction_settings):
        image_id = Path(result.path).stem
        h, w = result.orig_shape
        row = prediction_record(image_id, [w, h], result.boxes.cls.cpu().tolist(),
                                result.boxes.conf.cpu().tolist(), result.boxes.xyxy.cpu().tolist())
        records.append(row)
        failures[image_id] = inspection_failures(gt[image_id], row["detections"])
    if len(records) != len(images) or {r["image_id"] for r in records} != {r["image_id"] for r in images}:
        raise RuntimeError("prediction image pairing failed")
    write_json(output/"predictions.json", {"settings": {**prediction_settings, "stream": True},
                                         "preprocessing": "native RGB uint8 to normalized BCHW, fixed 1024 letterbox; library Results.xyxy restored to original pixels",
                                         "images": records})
    write_json(output/"failure-diagnostics.json", {"convention": "qualitative GT coverage at IoU>=0.5 and prediction confidence>=0.25, not one-to-one AP matching",
                                                "images": failures})
    inspected = {i: ["ordinary_preselected"] for i in selection["ordinary_val_ids"] if i in failures}
    if args.smoke and not inspected:
        inspected[images[0]["image_id"]] = ["smoke_reload"]
    for key, label in (("missed_weeds", "failure_missed_weeds"), ("class_confusions", "failure_crop_weed_confusion")):
        ranked = sorted(failures, key=lambda i: (-len(failures[i][key]), i))
        if ranked and failures[ranked[0]][key]:
            inspected.setdefault(ranked[0], []).append(label)
    (output/"overlays").mkdir()
    selected = []
    for record in records:
        image_id = record["image_id"]
        if image_id not in inspected:
            continue
        target = output/"overlays"/(image_id+".png")
        draw_overlay(derived/"images/val"/(image_id+".png"), target, record, gt[image_id], inspected[image_id])
        selected.append({**record, "selection": inspected[image_id], "failures": failures[image_id],
                         "overlay": str(target), "overlay_sha256": sha256_file(target)})
    write_json(output/"inspected-predictions.json", {"images": selected})
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--derived", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()

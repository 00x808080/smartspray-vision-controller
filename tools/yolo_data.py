#!/usr/bin/env python3
"""Deterministic YOLO export from existing audited half-open plant boxes."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import struct

try:
    from .audit_phenobench import json_text, sha256_file
except ImportError:
    from audit_phenobench import json_text, sha256_file

NAMES = {0: "crop", 1: "weed"}
POLICY = "all-annotation-valid-appearances-including-partial-v1"


def dimensions(size):
    if len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
        raise ValueError("size must contain positive integer width and height")
    return size


def xyxy_to_xywh(box, size):
    w, h = dimensions(size)
    if len(box) != 4 or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
        raise ValueError("bbox must contain four finite numbers")
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError("bbox must have positive extent within the image")
    return [(x0+x1)/(2*w), (y0+y1)/(2*h), (x1-x0)/w, (y1-y0)/h]


def xywh_to_xyxy(box, size):
    w, h = dimensions(size)
    if len(box) != 4 or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
        raise ValueError("normalized bbox must contain four finite numbers")
    cx, cy, bw, bh = box
    result = [(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h]
    xyxy_to_xywh(result, size)
    return result


def detector_class(class_id):
    if type(class_id) is not int or class_id not in (1, 2):
        raise ValueError("only adapter crop=1 and weed=2 are foreground")
    return class_id - 1


def prediction_record(image_id, size, classes, scores, boxes):
    """Serialize library Results.boxes.xyxy, already in original-image pixels."""
    dimensions(size)
    if not image_id or len(classes) != len(scores) or len(scores) != len(boxes):
        raise ValueError("invalid image ID or inconsistent prediction lengths")
    detections = []
    for cls, score, box in zip(classes, scores, boxes):
        if not isinstance(cls, (int, float)) or not math.isfinite(cls) or cls not in NAMES:
            raise ValueError("prediction class must be 0 or 1")
        if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("prediction score must be finite in [0, 1]")
        xyxy_to_xywh(box, size)  # Check coordinates without rescaling or clipping.
        detections.append({"class_id": int(cls), "class_name": NAMES[int(cls)],
                           "score": float(score), "xyxy": list(map(float, box))})
    return {"image_id": image_id, "width": size[0], "height": size[1], "detections": detections}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def ordinary_selection(images, per_date=2):
    """First IDs per acquisition date, fixed before any model output is read."""
    groups = defaultdict(list)
    for row in images:
        if row["split"] == "val":
            groups[row.get("acquisition_date") or "unknown"].append(row["image_id"])
    return sorted(image_id for date in sorted(groups) for image_id in sorted(groups[date])[:per_date])


def export_dataset(dataset_root, audit_root, output):
    root, audit, output = map(lambda p: Path(p).resolve(), (dataset_root, audit_root, output))
    if output.exists() or output == root or root in output.parents or output in root.parents:
        raise ValueError("output must be new and separate from the original dataset")
    if audit == output or audit in output.parents or output in audit.parents:
        raise ValueError("output must be separate from the audit")
    summary = json.loads((audit/"summary.json").read_text())
    if summary["technical_status"] != "checks_passed" or summary["issues"]["total"] != 0:
        raise ValueError("audit has unresolved invalid annotations")
    images = read_jsonl(audit/"images.jsonl")
    objects = read_jsonl(audit/"objects.jsonl")
    manifest = {r["path"]: r for r in read_jsonl(audit/"file_manifest.jsonl")}
    if (audit/"issues.jsonl").read_text().strip():
        raise ValueError("audit issues must be empty")
    image_map = {}
    for row in images:
        split, name, image_id = row["split"], row["filename"], row["image_id"]
        if split not in ("train", "val") or Path(name).name != name or name != image_id+".png":
            raise ValueError("invalid split, image ID, or filename")
        dimensions(row["size"])
        if row["status"] != "ok" or (split, image_id) in image_map:
            raise ValueError("invalid or duplicate image record")
        image_map[split, image_id] = row
    for split in ("train", "val"):
        folder = root/split/"images"
        if (root/split).is_symlink() or folder.is_symlink() or not folder.is_dir():
            raise ValueError("split/image directory must be a real directory")
        expected = {r["filename"] for r in images if r["split"] == split}
        actual = {p.name for p in folder.iterdir()}
        if expected != actual or len(expected) != summary["splits"][split]["images"]:
            raise ValueError("official split membership differs from the existing audit")
    grouped, keys = defaultdict(list), set()
    for obj in objects:
        key = obj["split"], obj["image_id"]
        cls = detector_class(obj["class_id"])
        raw_id = obj["raw_instance_id"]
        if type(raw_id) is not int or raw_id <= 0:
            raise ValueError("raw instance ID must be positive")
        obj_key = (*key, cls, raw_id)
        if key not in image_map or obj["filename"] != image_map[key]["filename"] or obj_key in keys:
            raise ValueError("unpaired or duplicate object record")
        if obj["annotation_valid"] is not True or obj["image_annotation_valid"] is not True:
            raise ValueError("invalid annotation must not be silently dropped")
        if obj["class_name"] != NAMES[cls] or type(obj["partial_semantic"]) is not bool:
            raise ValueError("inconsistent class/partial metadata")
        normalized = xyxy_to_xywh(obj["bbox"], image_map[key]["size"])
        restored = xywh_to_xyxy(normalized, image_map[key]["size"])
        if any(abs(a-b) > 1e-9 for a, b in zip(restored, obj["bbox"])):
            raise ValueError("bbox round trip failed")
        keys.add(obj_key)
        grouped[key].append((obj, cls, normalized))
    labels, counts, image_manifest = {}, {}, []
    for split in ("train", "val"):
        rows = sorted((r for r in images if r["split"] == split), key=lambda r: r["image_id"])
        class_counts, partial, max_objects = Counter(), 0, 0
        for row in rows:
            key = split, row["image_id"]
            records = sorted(grouped[key], key=lambda v: (v[1], v[0]["raw_instance_id"]))
            if len(records) != row["object_count"]:
                raise ValueError("per-image object count differs from audit")
            lines, native_rows = [], set()
            for obj, cls, normalized in records:
                line = str(cls)+" "+" ".join(format(v, ".17g") for v in normalized)
                # Native verification deduplicates float32 class+xywh rows. Fail
                # before training rather than silently change the annotation set.
                native = struct.pack("5f", cls, *normalized)
                if native in native_rows:
                    raise ValueError("framework would deduplicate equal float32 labels")
                native_rows.add(native)
                lines.append(line)
                class_counts[NAMES[cls]] += 1
                partial += obj["partial_semantic"]
            label = "\n".join(lines) + ("\n" if lines else "")
            labels[key] = label
            source = root/split/"images"/row["filename"]
            expected = manifest[f"{split}/images/{row['filename']}"]
            if source.is_symlink() or not source.is_file() or expected["status"] != "readable":
                raise ValueError("RGB source must be an audited regular file")
            if expected["size"] != row["size"] or source.stat().st_size != expected["bytes"] or sha256_file(source) != expected["sha256"]:
                raise ValueError("RGB file changed since audit")
            max_objects = max(max_objects, len(records))
            import hashlib
            image_manifest.append({**row, "rgb_sha256": expected["sha256"],
                                   "label_sha256": hashlib.sha256(label.encode()).hexdigest()})
        old = summary["splits"][split]["objects"]
        if any(class_counts[name] != old[name] for name in NAMES.values()) or partial != old["partial_semantic"]:
            raise ValueError("split class/partial counts differ from audit")
        counts[split] = {"images": len(rows), "objects": sum(class_counts.values()),
                         "classes": dict(class_counts), "partial": partial, "max_objects": max_objects}
    output.mkdir(parents=True)
    for split in ("train", "val"):
        (output/"images"/split).mkdir(parents=True)
        (output/"labels"/split).mkdir(parents=True)
    for row in image_manifest:
        split, image_id = row["split"], row["image_id"]
        (output/"images"/split/row["filename"]).symlink_to(root/split/"images"/row["filename"])
        (output/"labels"/split/(image_id+".txt")).write_text(labels[split, image_id])
    (output/"images.jsonl").write_text("".join(json_text(r) for r in image_manifest))
    # Keep raw IDs and all partial/visibility metadata in local provenance only.
    ordered_objects = sorted(objects, key=lambda r: (("train", "val").index(r["split"]), r["image_id"], r["class_id"], r["raw_instance_id"]))
    (output/"provenance.jsonl").write_text("".join(json_text(r) for r in ordered_objects))
    full = f"path: {json.dumps(str(output))}\ntrain: images/train\nval: images/val\nnames:\n  0: crop\n  1: weed\n"
    (output/"dataset.yaml").write_text(full)
    subset = {}
    for split, limit in (("train", 32), ("val", 16)):
        ids = [r["image_id"] for r in image_manifest if r["split"] == split][:limit]
        subset[split] = ids
        (output/f"smoke-{split}.txt").write_text("".join(f"./images/{split}/{i}.png\n" for i in ids))
    (output/"smoke.yaml").write_text(full.replace("train: images/train", "train: smoke-train.txt").replace("val: images/val", "val: smoke-val.txt"))
    (output/"selection.json").write_text(json_text({"ordinary_val_ids": ordinary_selection(images), "smoke": subset,
                                                   "selection_rule": "first two sorted validation IDs per acquisition date; frozen before predictions"}))
    report = {"schema_version": 1, "policy": POLICY, "names": NAMES, "counts": counts,
              "framework_duplicate_rows": 0, "source_audit_sha256": {name: sha256_file(audit/name) for name in ("summary.json", "images.jsonl", "objects.jsonl", "file_manifest.jsonl", "issues.jsonl")}}
    (output/"export.json").write_text(json_text(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--audit-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json_text(export_dataset(args.dataset_root, args.audit_root, args.output)), end="")


if __name__ == "__main__":
    main()

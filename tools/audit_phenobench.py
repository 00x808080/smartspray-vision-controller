#!/usr/bin/env python3
"""Audit official PhenoBench train/val and export ground-truth plant boxes."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image, ImageDraw

try:
    from .phenobench_data import extract_objects
except ImportError:  # Direct script invocation from the repository root.
    from phenobench_data import extract_objects

SPLITS = ("train", "val")
KINDS = ("images", "semantics", "plant_instances", "plant_visibility", "leaf_instances", "leaf_visibility")
BBOX_CONVENTION = "[x_min, y_min, x_max_exclusive, y_max_exclusive]"


def json_text(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_png(path):
    """Load without mode conversion, retaining full integer mask values."""
    with Path(path).open("rb") as stream:
        header = stream.read(29)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError("not a PNG with an IHDR header")
    with Image.open(path) as image:
        image.load()
        array = np.array(image)
        info = {"size": list(image.size), "mode": image.mode, "bit_depth": header[24], "color_type": header[25], "dtype": str(array.dtype)}
    return array, info


def filename_metadata(name):
    match = re.fullmatch(r"(\d{2}-\d{2})_(\d+)_([^./]+)\.png", name)
    if match is None:
        return {"acquisition_date": None, "source_group_candidate": None}
    date, patch, source = match.groups()
    return {"acquisition_date": date, "patch_token": patch, "source_token": source,
            "source_group_candidate": date + "/" + source}


def distribution(values):
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    # Nearest rank on [0,n-1], no version-dependent floating interpolation.
    return {"count": len(ordered), "min": ordered[0], "p25": ordered[(len(ordered)-1)//4],
            "median": ordered[(len(ordered)-1)//2], "p75": ordered[3*(len(ordered)-1)//4], "max": ordered[-1]}


def _preview_selection(images, limit):
    groups = defaultdict(list)
    for row in images:
        if row["split"] == "train" and row["status"] == "ok" and row["object_count"]:
            groups[row["acquisition_date"] or "unknown"].append(row)
    candidates = []
    # Round-robin dates: smallest present object, largest present object,
    # median typical object size, then earliest remaining image ID.
    for criterion in range(4):
        for date in sorted(groups):
            rows = groups[date]
            if criterion == 0:
                ranked = sorted(rows, key=lambda r: (r["min_bbox_area_px"], r["image_id"]))
            elif criterion == 1:
                ranked = sorted(rows, key=lambda r: (-r["max_bbox_area_px"], r["image_id"]))
            elif criterion == 2:
                ordered = sorted(rows, key=lambda r: (r["median_bbox_area_px"], r["image_id"]))
                middle = len(ordered)//2
                ranked = sorted(ordered, key=lambda r: (abs(r["median_bbox_area_px"]-ordered[middle]["median_bbox_area_px"]), r["image_id"]))
            else:
                ranked = sorted(rows, key=lambda r: r["image_id"])
            for row in ranked:
                if row not in candidates:
                    candidates.append(row)
                    break
    return candidates[:limit]


def make_previews(root, output, images, limit):
    selected = _preview_selection(images, limit)
    if not selected:
        return []
    selected_ids = {row["image_id"] for row in selected}
    records = defaultdict(list)
    with (output / "objects.jsonl").open() as stream:
        for line in stream:
            record = json.loads(line)
            if record["split"] == "train" and record["image_id"] in selected_ids:
                records[record["image_id"]].append(record)
    folder = output / "previews"
    folder.mkdir()
    manifest = []
    for row in selected:
        with Image.open(root / "train/images" / row["filename"]) as source:
            source.load()
            # Conversion is only for the RGB preview, never for annotation masks.
            image = source.convert("RGB")
        canvas = Image.new("RGB", (image.width, image.height + 48), "#111111")
        canvas.paste(image, (0, 48))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 5), "GROUND-TRUTH ANNOTATION PREVIEW - NOT PREDICTIONS", fill="white")
        draw.text((8, 23), "train / " + row["filename"] + " | green=crop red=weed yellow=partial/ignore", fill="white")
        for obj in records[row["image_id"]]:
            x0, y0, x1, y1 = obj["bbox"]
            partial = obj["partial_semantic"] or obj["official_ignore_candidate"]
            color = "yellow" if partial else ("lime" if obj["class_id"] == 1 else "red")
            if not obj["annotation_valid"]:
                color = "magenta"
            draw.rectangle((x0, y0+48, x1-1, y1-1+48), outline=color, width=1)
            label = obj["class_name"] + ":" + str(obj["raw_instance_id"]) + (" partial" if partial else "")
            draw.text((x0, max(48, y0+38)), label, fill=color, stroke_width=1, stroke_fill="black")
        target = folder / (row["image_id"] + ".png")
        canvas.save(target)
        manifest.append({"split": "train", "image_id": row["image_id"], "acquisition_date": row["acquisition_date"],
                         "min_bbox_area_px": row["min_bbox_area_px"], "max_bbox_area_px": row["max_bbox_area_px"],
                         "file": "previews/"+target.name, "sha256": sha256_file(target)})
    return manifest


def audit_dataset(root, output, previews=0):
    """Inspect only train/val; refuse overwriting an existing output directory.

    Detailed artifacts are local audit records, not a training dataset. Any
    error leaves an explicit issue and marks the report blocked. The report
    deliberately makes no automatic training-readiness or license decision.
    """
    root, output = Path(root).resolve(), Path(output).resolve()
    if not root.is_dir():
        raise ValueError("dataset root must be an existing directory")
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("output and source dataset directories must be separate")
    if not 0 <= previews <= 12:
        raise ValueError("previews must be between 0 and 12")
    output.mkdir(parents=True, exist_ok=False)
    hashes = defaultdict(list)
    issues = []
    images = []
    inventories = {}
    split_stats = {}
    group_sets = {split: set() for split in SPLITS}
    instance_sets = {split: {1: set(), 2: set()} for split in SPLITS}
    date_counts = {split: Counter() for split in SPLITS}
    bbox_values = {split: {class_id: defaultdict(list) for class_id in (1, 2)} for split in SPLITS}
    object_counts = {split: Counter() for split in SPLITS}

    def issue(code, split=None, filename=None, **details):
        item = {"code": code, **details}
        if split is not None:
            item["split"] = split
        if filename is not None:
            item["filename"] = filename
        issues.append(item)

    with (output / "file_manifest.jsonl").open("w") as file_stream, (output / "objects.jsonl").open("w") as object_stream:
        for split in SPLITS:
            split_is_symlink = (root / split).is_symlink()
            if split_is_symlink:
                issue("symlink_split_directory", split)
            inventories[split] = {}
            split_stats[split] = {}
            for kind in KINDS:
                folder = root / split / kind
                names = []
                if split_is_symlink or folder.is_symlink() or not folder.is_dir():
                    issue("missing_or_symlink_directory", split, directory=kind)
                else:
                    for path in sorted(folder.iterdir()):
                        if path.is_symlink() or not path.is_file() or path.suffix != ".png":
                            issue("unsupported_directory_entry", split, path.name, directory=kind)
                        else:
                            names.append(path.name)
                inventories[split][kind] = set(names)
                split_stats[split][kind] = {"files": len(names), "readable": 0, "corrupt": 0,
                                          "dimensions": Counter(), "modes": Counter(), "bit_depths": Counter(), "values": Counter()}
            all_names = sorted(set().union(*inventories[split].values()))
            if not all_names:
                issue("empty_split", split)
            for name in all_names:
                initial_issues = len(issues)
                metadata = filename_metadata(name)
                if name in inventories[split]["images"]:
                    date_counts[split][metadata["acquisition_date"] or "unknown"] += 1
                    if metadata["source_group_candidate"]:
                        group_sets[split].add(metadata["source_group_candidate"])
                    else:
                        issue("unrecognized_filename_metadata", split, name)
                arrays, infos = {}, {}
                for kind in KINDS:
                    if name not in inventories[split][kind]:
                        issue("missing_file", split, name, directory=kind)
                        continue
                    path = root / split / kind / name
                    relative = f"{split}/{kind}/{name}"
                    stats = split_stats[split][kind]
                    try:
                        digest = sha256_file(path)
                        hashes[digest].append(relative)
                        manifest = {"path": relative, "bytes": path.stat().st_size, "sha256": digest}
                        try:
                            array, info = load_png(path)
                        except (OSError, ValueError, SyntaxError) as exc:
                            stats["corrupt"] += 1
                            manifest["status"] = "unreadable"
                            issue("unreadable_png", split, name, directory=kind, error_type=type(exc).__name__)
                        else:
                            arrays[kind], infos[kind] = array, info
                            stats["readable"] += 1
                            stats["dimensions"]["x".join(map(str, info["size"]))] += 1
                            stats["modes"][info["mode"]] += 1
                            stats["bit_depths"][str(info["bit_depth"])] += 1
                            manifest.update(info, status="readable")
                            if kind == "images":
                                if array.ndim != 3 or array.shape[2] != 3 or info["mode"] != "RGB":
                                    issue("unsupported_image_format", split, name, mode=info["mode"])
                            else:
                                if array.ndim != 2 or array.dtype.kind not in "ui":
                                    issue("unsupported_mask_format", split, name, directory=kind)
                                else:
                                    values, counts = np.unique(array, return_counts=True)
                                    stats["values"].update({str(int(v)): int(c) for v, c in zip(values, counts)})
                                    maximum = 4 if kind == "semantics" else (255 if "visibility" in kind else 65535)
                                    if int(values.min()) < 0 or int(values.max()) > maximum:
                                        issue("unsupported_mask_values", split, name, directory=kind, minimum=int(values.min()), maximum=int(values.max()))
                                    allowed_depths = (8, 16) if "visibility" in kind else (16,)
                                    if info["bit_depth"] not in allowed_depths:
                                        issue("unexpected_mask_bit_depth", split, name, directory=kind, bit_depth=info["bit_depth"])
                        file_stream.write(json_text(manifest))
                    except OSError as exc:
                        issue("file_io_error", split, name, directory=kind, error_type=type(exc).__name__)
                objects = []
                if "images" in infos:
                    size = infos["images"]["size"]
                    for kind in KINDS[1:]:
                        if kind in infos and infos[kind]["size"] != size:
                            issue("dimension_mismatch", split, name, directory=kind, image_size=size, mask_size=infos[kind]["size"])
                    if len(infos) == len(KINDS) and all(info["size"] == size for info in infos.values()):
                        try:
                            objects, object_issues = extract_objects(arrays["semantics"], arrays["plant_instances"], arrays["plant_visibility"])
                        except ValueError as exc:
                            issue("invalid_annotation", split, name, detail=str(exc))
                        else:
                            for detail in object_issues:
                                detail = dict(detail)
                                code = detail.pop("code")
                                issue(code, split, name, **detail)
                for obj in objects:
                    record = {"image_id": Path(name).stem, "filename": name, "split": split, "image_annotation_valid": len(issues) == initial_issues, **metadata, **obj}
                    object_stream.write(json_text(record))
                    class_id = obj["class_id"]
                    counts = object_counts[split]
                    counts[obj["class_name"]] += 1
                    counts["partial_semantic"] += int(obj["partial_semantic"])
                    counts["official_ignore_candidate"] += int(obj["official_ignore_candidate"] is True)
                    counts["edge_touching"] += int(obj["touches_image_edge"])
                    counts["invalid_objects"] += int(not obj["annotation_valid"])
                    instance_sets[split][class_id].add(obj["raw_instance_id"])
                    x0, y0, x1, y1 = obj["bbox"]
                    for metric, value in [("width_px", x1-x0), ("height_px", y1-y0), ("bbox_area_px", (x1-x0)*(y1-y0)), ("mask_area_px", obj["mask_area_px"])]:
                        bbox_values[split][class_id][metric].append(value)
                areas = sorted((o["bbox"][2]-o["bbox"][0])*(o["bbox"][3]-o["bbox"][1]) for o in objects)
                images.append({"split": split, "image_id": Path(name).stem, "filename": name, **metadata,
                               "size": infos.get("images", {}).get("size"), "object_count": len(objects),
                               "status": "ok" if len(issues) == initial_issues else "review_required",
                               "min_bbox_area_px": min(areas) if areas else None,
                               "max_bbox_area_px": max(areas) if areas else None,
                               "median_bbox_area_px": areas[(len(areas)-1)//2] if areas else None})
    duplicates = [{"sha256": digest, "paths": paths} for digest, paths in sorted(hashes.items()) if len(paths) > 1]
    cross_split_duplicates = [group for group in duplicates if len({p.split("/")[0] for p in group["paths"]}) > 1]
    image_duplicates = [group for group in cross_split_duplicates if len({p.split("/")[0] for p in group["paths"] if p.split("/")[1] == "images"}) > 1]
    report_splits = {}
    for split in SPLITS:
        by_kind = {}
        for kind, stats in split_stats[split].items():
            values = stats.pop("values")
            value_ids = sorted(map(int, values))
            stats["value_summary"] = {"unique_count": len(value_ids), "min": min(value_ids) if value_ids else None, "max": max(value_ids) if value_ids else None}
            if kind != "images":
                stats["value_pixel_counts"] = values
            by_kind[kind] = stats
        report_splits[split] = {"files": by_kind, "dates": date_counts[split], "objects": object_counts[split],
                               "source_group_candidates": len(group_sets[split]),
                               "unique_crop_ids": len(instance_sets[split][1]), "unique_weed_numeric_ids": len(instance_sets[split][2]),
                               "bbox_distributions": {"crop" if k == 1 else "weed": {m: distribution(v) for m, v in metrics.items()} for k, metrics in bbox_values[split].items()}}
    report = {
        "schema_version": 1,
        "bbox_convention": BBOX_CONVENTION,
        "audited_splits": list(SPLITS),
        "hidden_test": "not opened or used for decisions",
        "training_policy": "undecided; preserve every partial/ignore region; consult docs/DATA.md",
        "technical_status": "issues_require_review" if issues else "checks_passed",
        "ready_for_training": False,
        "rights_status": "not inferred by this tool; consult local provenance and docs/DATA.md",
        "splits": report_splits,
        "issues": {"total": len(issues), "by_code": Counter(i["code"] for i in issues)},
        "exact_duplicate_groups": duplicates,
        "cross_split_duplicate_groups": cross_split_duplicates,
        "cross_split_image_duplicate_groups": image_duplicates,
        "split_overlap": {
            "crop_ids": sorted(instance_sets["train"][1] & instance_sets["val"][1]),
            "crop_id_scope": "paper documents globally aligned crop identities; numeric intersections are observed evidence",
            "weed_numeric_ids": sorted(instance_sets["train"][2] & instance_sets["val"][2]),
            "weed_id_scope": "numeric overlap only; global physical identity not established",
            "source_group_candidates": sorted(group_sets["train"] & group_sets["val"]),
            "group_scope": "inferred date/source filename tokens, not verified physical groups",
            "independence": "file hashes do not establish spatial or temporal independence",
        },
        "root_metadata_files": sorted(p.name for p in root.iterdir() if p.is_file()),
    }
    (output / "images.jsonl").write_text("".join(json_text(row) for row in images))
    (output / "issues.jsonl").write_text("".join(json_text(row) for row in issues))
    preview_manifest = make_previews(root, output, images, previews)
    (output / "previews.json").write_text(json_text(preview_manifest))
    report["preview_count"] = len(preview_manifest)
    (output / "report.json").write_text(json_text(report))
    summary = {
        "technical_status": report["technical_status"], "issues": report["issues"],
        "splits": {split: {"images": report_splits[split]["files"]["images"]["files"], "objects": report_splits[split]["objects"]} for split in SPLITS},
        "cross_split_crop_ids": len(report["split_overlap"]["crop_ids"]),
        "cross_split_image_duplicate_groups": len(image_duplicates), "previews": len(preview_manifest),
        "ready_for_training": False,
    }
    (output / "summary.json").write_text(json_text(summary))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="directory containing official train/ and val/")
    parser.add_argument("--output", type=Path, required=True, help="new local directory outside the dataset")
    parser.add_argument("--previews", type=int, choices=range(13), default=0, metavar="0..12")
    args = parser.parse_args(argv)
    try:
        report = audit_dataset(args.dataset_root, args.output, args.previews)
    except (OSError, ValueError) as exc:
        print(f"Audit failed: {exc}", file=sys.stderr)
        return 1
    print((args.output / "summary.json").read_text(), end="")
    return 2 if report["issues"]["total"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
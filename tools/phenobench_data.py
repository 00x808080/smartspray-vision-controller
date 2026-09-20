"""Pure PhenoBench plant-mask conversion; no filesystem or training code."""

import numpy as np


LABELS = {0: "background", 1: "crop", 2: "weed", 3: "partial_crop", 4: "partial_weed"}


def _mask(array, name, maximum):
    array = np.asarray(array)
    if array.ndim != 2 or not all(array.shape):
        raise ValueError(f"{name}: expected a nonempty two-dimensional mask")
    if array.dtype.kind not in "ui" or np.any(array < 0) or np.any(array > maximum):
        raise ValueError(f"{name}: expected integer values in [0, {maximum}]")
    return array


def extract_objects(semantics, instances, visibility=None):
    """Return (objects, issues), sorted by (class_id, raw_instance_id).

    Crop/weed IDs have separate namespaces. Partial and regular labels for the
    same class/ID are grouped, with contradictions reported, never voted away.
    All known-class positive-ID regions are retained; annotation_valid=False
    explicitly quarantines contradictory objects for any future training use.
    Invalid mask shape/dtype/range raises ValueError for the whole image.
    Inputs are never modified. Bboxes are half-open original-pixel extents.
    """
    semantics = _mask(semantics, "semantics", 4)
    instances = _mask(instances, "plant_instances", 65535)
    if semantics.shape != instances.shape:
        raise ValueError("semantics and plant_instances dimensions differ")
    if visibility is not None:
        visibility = _mask(visibility, "plant_visibility", 255)
        if visibility.shape != semantics.shape:
            raise ValueError("plant_visibility dimensions differ")
    height, width = semantics.shape
    issues = []
    for code, mask in [
        ("foreground_without_instance", (semantics > 0) & (instances == 0)),
        ("instance_on_background", (semantics == 0) & (instances > 0)),
    ]:
        count = int(np.count_nonzero(mask))
        if count:
            issues.append({"code": code, "pixels": count})
    background_instance_ids = set(map(int, np.unique(instances[(semantics == 0) & (instances > 0)])))
    ys, xs = np.nonzero((semantics > 0) & (instances > 0))
    if not len(xs):
        return [], issues
    raw_labels = semantics[ys, xs]
    classes = np.where((raw_labels == 1) | (raw_labels == 3), 1, 2)
    # Explicit widening before constructing keys avoids uint16 overflow.
    keys = classes.astype(np.int64) * 65536 + instances[ys, xs].astype(np.int64)
    order = np.argsort(keys, kind="stable")
    keys, ys, xs = keys[order], ys[order], xs[order]
    starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1, len(keys)]
    objects = []
    for start, stop in zip(starts[:-1], starts[1:]):
        y, x = ys[start:stop], xs[start:stop]
        class_id, raw_id = divmod(int(keys[start]), 65536)
        labels = np.unique(semantics[y, x]).astype(int).tolist()
        partial = any(label in (3, 4) for label in labels)
        problems = []
        if raw_id in background_instance_ids:
            problems.append("instance_id_also_on_background")
        if len(labels) != 1:
            problems.append("mixed_regular_partial_labels")
        values, counts = ([], []) if visibility is None else np.unique(visibility[y, x], return_counts=True)
        values = [int(v) for v in values]
        counts = [int(v) for v in counts]
        ignore = None if not values else all(v / 255 <= 0.5 for v in values)
        if len(values) > 1:
            problems.append("nonuniform_instance_visibility")
            if min(values) <= 127 < max(values):
                ignore = None
        if len(labels) == 1 and ignore is not None and partial != ignore:
            problems.append("partial_label_visibility_disagreement")
        bbox = [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1]
        obj = {
            "raw_instance_id": raw_id,
            "class_id": class_id,
            "class_name": "crop" if class_id == 1 else "weed",
            "semantic_labels": labels,
            "bbox": bbox,
            "mask_area_px": int(stop - start),
            "partial_semantic": partial,
            "visibility_values": values,
            "visibility_pixel_counts": counts,
            "official_ignore_candidate": ignore,
            "touches_image_edge": bbox[0] == 0 or bbox[1] == 0 or bbox[2] == width or bbox[3] == height,
            "annotation_valid": not problems,
        }
        objects.append(obj)
        for code in problems:
            issues.append({"code": code, "class_id": class_id, "raw_instance_id": raw_id})
    return objects, issues
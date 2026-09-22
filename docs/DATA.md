# PhenoBench data audit and annotation adapter (M2.1)

## Scope and status

M2.1 extracts ground-truth **crop** and **weed** objects from existing plant annotations. It does not train a model, run inference, create controller commands, or implement Python/C++ integration. M2 is not complete. Training readiness is blocked by the conflicting license notices described below; technical findings must also be resolved before baseline selection.

The proposed later simulation anchor is the center of a detected weed box in original-image pixels. It is not a verified stem location. No anchor or controller target is produced by this audit.

## Provenance and rights

The audit candidate is the official [PhenoBench-v110.zip](https://www.phenobench.org/data/PhenoBench-v110.zip), version **1.1.0 (December 2023)** according to its README, retrieved **2026-09-20 UTC** from the [publisher's dataset page](https://www.phenobench.org/dataset.html).

| Property | Recorded value |
|---|---|
| Archive size (HTTP and local file) | 7,630,658,167 bytes |
| Publisher MD5; independently verified | `5168bba762053725890478432cdbdb1d` |
| Independently calculated SHA256 | `a310c1c1dcfec4ca4897097fd38c02af39ea24112f658fd9408d5deb3c7c8bfc` |
| ZIP advertised total extracted size | 7,664,798,921 bytes |
| Actually extracted train/val plus README | 5,799,608,616 bytes; 13,075 files |

The publisher supplies an MD5, not a published SHA256. The SHA256 above identifies the independently downloaded package; it is not a publisher signature. ZIP member paths, duplicate names, symlinks, encryption flags, and expected disk usage were checked before extraction. Standard `curl` and Python `zipfile` were used. Only labelled train/val and the README were extracted; hidden test content remains unopened inside the original archive. No unlabelled datasets, model weights, or predictions were downloaded.

**Unresolved license conflict:** the current website and [official FAQ](https://github.com/PRBonn/phenobench/blob/0edc128ef7f67c8c6577554c7d1a2e382e2ea81f/README.md#frequently-asked-questions) state **CC BY-SA 4.0**. The original archive's `PhenoBench/README.MD` instead states **CC BY-NC-SA 4.0**, including a noncommercial restriction. Both statements are retained locally; neither is silently substituted for the other. This audit does not establish publication or commercial-use rights. Images, derived previews, full object records, and detailed reports stay local. No authors were contacted. Dataset, model weights, model code, inference runtime, and this project's own code require separate rights decisions; the devkit's MIT license does not resolve the data conflict.

Local acquisition evidence includes the retrieved webpages, pinned devkit source, original archive notice, HTTP metadata, central directory, calculated checksums, extraction record, and evidence-file hashes. These are intentionally excluded from Git.

## Publisher-reported properties and verified conventions

The [dataset description](https://www.phenobench.org/dataset.html) reports 1,407 training and 772 validation images, 1024 x 1024 pixels, five 16-bit PNG annotation layers, and date prefixes in filenames. The [paper, sections III-A/C/D](https://arxiv.org/html/2306.04557v2) describes overlapping patches, spatial row-based splits, and temporal alignment of **crop** identities. Global physical identity of weed and leaf IDs is not established by that crop-specific statement.

The inspected [devkit](https://github.com/PRBonn/phenobench/tree/0edc128ef7f67c8c6577554c7d1a2e382e2ea81f) is pinned to commit `0edc128ef7f67c8c6577554c7d1a2e382e2ea81f`. Its [loader](https://github.com/PRBonn/phenobench/blob/0edc128ef7f67c8c6577554c7d1a2e382e2ea81f/src/phenobench/phenobench_loader.py) and [conversion code](https://github.com/PRBonn/phenobench/blob/0edc128ef7f67c8c6577554c7d1a2e382e2ea81f/src/phenobench/evaluation/auxiliary/convert.py) resolve the repeated `partial-crop` typo for label 4 on the webpage and archive README:

| Raw semantic label | Meaning | Adapter foreground class |
|---|---|---|
| 0 | Background | None |
| 1 | Crop | crop (1) |
| 2 | Weed | weed (2) |
| 3 | Partial crop | crop (1), partial retained |
| 4 | Partial weed | weed (2), partial retained |

Instance 0 denotes background; positive plant IDs are preserved without 8-bit conversion. Crop and weed have separate ID namespaces, following the loader's explicit distinction. An object key is `(split, image_id, class_id, raw_instance_id)`. Disconnected pixels with the same class/ID form one box. The archive's date-prefix meaning is documented; `date/source-token` grouping from other filename fields is an **inferred candidate**, not verified acquisition geometry.

The publisher describes partial plants as less than 50% visible. The [official detection filter](https://github.com/PRBonn/phenobench/blob/0edc128ef7f67c8c6577554c7d1a2e382e2ea81f/src/phenobench/evaluation/auxiliary/filter.py) uses normalized visibility **<= 0.5**, and removes a prediction when overlap with a partial GT box exceeds half the prediction's area, independently of class. That overlap ratio is not IoU. Its GT removal is nested inside the prediction loop, so zero predictions are an upstream edge case. The [evaluator](https://github.com/PRBonn/phenobench/blob/0edc128ef7f67c8c6577554c7d1a2e382e2ea81f/src/phenobench/evaluation/evaluate_plant_bounding_boxes.py) calls this filter before mAP. This project records ignore candidates but does not implement or claim equivalence with that evaluator.

## Adopted adapter choices

`tools/phenobench_data.py` contains reusable array conversion; `tools/audit_phenobench.py` handles files, reports, and local previews. Both use only the standard library, NumPy, and Pillow.

- Coordinates are original-image pixel extents: **`[x_min, y_min, x_max_exclusive, y_max_exclusive]`**. Maximal occupied indices receive `+1`. A one-pixel object therefore has width and height 1. This intentionally differs from the upstream `max-min` and rounded-center boxes; benchmark equivalence is not claimed.
- Every known foreground class/positive ID retains its raw semantic labels, pixel count, edge-touch flag, visibility values and their pixel counts, semantic partial flag, and visibility-based ignore hint. Edge contact does not itself imply partial visibility.
- Mixed regular/partial labels or nonuniform visibility remain explicit findings. Contradictory objects retain their record with `annotation_valid=false`; `image_annotation_valid` also carries file/image-level findings; no majority vote or mask repair is performed. Orphan foreground pixels and instances on background are reported. Unsupported labels, shapes, or ranges quarantine image conversion with an explicit finding.
- Future training treatment of partial/ignore regions is **undecided**. They must not silently disappear or become negatives. `official_ignore_candidate` is only a visibility hint; it is not a training policy or complete evaluation ignore implementation.
- Deterministic order is train then val, sorted filename, class ID, raw ID. Source arrays and files are not modified. An empty valid annotation produces zero objects; an unreadable or invalid annotation is reported separately.

The audit inventories, hashes, and decodes every available PNG in the six required directories of each labelled split: `images`, `semantics`, `plant_instances`, `plant_visibility`, `leaf_instances`, `leaf_visibility`. It checks missing/orphan pairing, dimensions, mask values/bit depth, file readability, object counts and size distributions, metadata, exact duplicate files, candidate source groups, and crop-ID intersections. Leaf layers are checked as files/masks; this task does not create leaf objects or validate a leaf segmentation model.

Output records are audit artifacts, not a finalized training format. A missing required layer prevents that image's conversion and is reported. `issues.jsonl` identifies affected images/objects, and `images.jsonl` distinguishes `ok` from `review_required`. Exit status is 0 for no technical findings, 2 for a completed audit with findings, and 1 for invocation/output failure. License readiness is never inferred from exit status.

## Measured package results

The complete available train/val package contains **13,074 PNG files**, all readable, all 1024 x 1024, with complete matching six-directory membership. The root README is the only packaged metadata file outside those images/masks. No corrupt files, pairing errors, unsupported semantic values, conflicting plant classes/visibility, or dimension mismatches were found.

| Measurement | Train | Validation |
|---|---:|---:|
| Images (and files per annotation layer) | 1,407 | 772 |
| Crop instance appearances | 11,875 | 6,482 |
| Weed instance appearances | 8,141 | 3,926 |
| Partial / visibility-ignore appearances, both classes | 3,566 | 1,840 |
| Boxes touching image edge | 6,546 | 3,383 |
| Distinct crop IDs | 566 | 554 |
| Distinct weed numeric IDs | 756 | 680 |
| Inferred date/source groups | 21 | 12 |
| Maximum raw plant / leaf ID | 1,559 / 1,874 | 2,027 / 1,532 |

Counts are per-image appearances, not distinct physical plants. Crop/weed class namespaces are retained. Semantic values are exactly `{0,1,2,3,4}`. RGB images are 8-bit per channel. Semantics and both instance layers are **16-bit**; both visibility layers are actually **8-bit**, spanning 0..255. This differs from the publisher's blanket 16-bit statement but represents the documented visibility range without loss. The adapter accepts 8/16-bit visibility and requires 16-bit stored instance/semantic masks.

Acquisition-date image counts are train/val: `05-15` = 613/399, `05-26` = 396/170, `06-05` = 398/203. Dates refer to 2020 according to the packaged README. No georeferenced patch footprints, camera poses, or independently validated physical source-group mapping are provided in this archive.

| Bbox area in pixels, min / median / max | Train | Validation |
|---|---|---|
| Crop | 1 / 13,005 / 419,584 | 1 / 9,563 / 449,341 |
| Weed | 1 / 900 / 257,103 | 2 / 1,173 / 245,178 |

The detailed report also contains width, height, mask-area and quartile distributions. One-pixel crop/weed extents are retained rather than becoming zero-sized boxes.

**Split findings:** five exact duplicate-file groups are matching plant/leaf visibility files for the same image; none cross splits, and no cross-split exact image duplicates were found. The inferred date/source tokens have zero train/val intersection. However, **537 crop IDs occur in both train and val**; the paper's global crop-identity scope makes this material overlap evidence. A further 258 weed numbers intersect, but physical weed identity is not established by their numerical equality. Official membership is preserved; no independent held-out claim is made. A suitable final evaluation protocol remains unresolved.

**Verification:** 36 offline synthetic unittest cases pass. Two complete real-data runs return exit 0 with zero structural/annotation issues; all **19 artifacts match byte-for-byte**, including 12 preview PNGs. All 12 training previews (four per date) were visually inspected; no material rendering error was found. Some border labels are cosmetically clipped. This is a plausibility review, not independent botanical annotation validation. Existing Debug and Release CTest each pass 1 registered test / **40 scenarios**. The C++ sources, tests, build configuration, and original Windows copy remain unchanged.

## Splits and planned evaluation

Official train/val membership is preserved. Do not repartition overlapping images randomly or split augmentations independently of source images. Global crop IDs can support crop overlap checks; repeated weed/leaf numbers alone do not prove shared physical plants. Exact file hashes establish byte identity only, not spatial or temporal independence. Filename group candidates do not replace verified source geometry. The hidden test set is neither opened for development nor used to select examples.

Visual inspection and demo candidates use training images only. Preview selection is deterministic: round-robin acquisition dates, images containing the smallest/largest boxes, a median typical object size, then the first remaining image ID, capped at 12. Every preview is marked **GROUND-TRUTH ANNOTATION PREVIEW - NOT PREDICTIONS**; it is not inference output.

The planned main metric is detection **AP averaged over IoU thresholds 0.50:0.05:0.95**, including separate crop and weed results; AP50/AP75 can be secondary diagnostics. Evaluation implementation, box and partial conventions must be fixed before comparisons. Validation used for tuning is not an independent final test. Numerical targets, baseline/model choice, weight provenance, and a final evaluation protocol remain a next decision **before final-model evaluation**. No model score is claimed here.

## Reproduction

Verified environment: Ubuntu 24.04 / WSL 2, existing CPython 3.12.3, uv 0.12.17, NumPy 2.5.3, Pillow 12.3.0. The isolated environment uses the existing system Python; it requires no OS package installation or replacement Python. Install uv according to its [official instructions](https://docs.astral.sh/uv/getting-started/installation/) if unavailable. The exact environment uses only the two packages in `requirements-data.txt`.

From the repository root, with `uv` available (the audited local installation is `.local/m2.1/uv/uv`):

```bash
uv venv --python /usr/bin/python3 --no-python-downloads .venv
uv pip install --python .venv/bin/python --no-python-downloads --only-binary :all: -r requirements-data.txt
.venv/bin/python -m unittest discover -s tests -p 'test_phenobench_data.py' -v
```

Create `.venv` only when absent; reuse a compatible existing environment rather than replacing it. Use the unchanged official package after validating its checksum, archive paths, size, and current rights. The data root is explicitly supplied and must contain `train/` and `val/`:

```bash
.venv/bin/python tools/audit_phenobench.py --dataset-root /path/to/PhenoBench --output .local/m2.1/run-a --previews 12
.venv/bin/python tools/audit_phenobench.py --dataset-root /path/to/PhenoBench --output .local/m2.1/run-b --previews 12
```

Output directories must be new and separate from the dataset. Compare the bytes of `report.json`, `summary.json`, `file_manifest.jsonl`, `images.jsonl`, `objects.jsonl`, `issues.jsonl`, `previews.json`, and preview PNGs across runs. Substantive artifacts contain no wall-clock timing or absolute source paths; acquisition dates and runtime logs are kept separately. Normal tests create only small temporary arrays/files and never download or require the real dataset.

The detailed reports, annotations, images, previews, `.venv`, and `.local` remain outside version control. The C++ contract and implementation remain unchanged; CTest regression is recorded separately from Python and dataset checks.
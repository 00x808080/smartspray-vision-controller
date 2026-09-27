# YOLO11n crop/weed detection baseline (M2.2)

One local educational portfolio baseline, initialized from official COCO-pretrained YOLO11n weights. The original controller is unchanged. This stage produces detections in original-image pixels; it does not export ONNX, create metric-space targets, invoke C++, or integrate hardware.

## Model, environment and provenance

| Item | Tested value |
|---|---|
| Model | Ultralytics YOLO11n detection, 2 classes; 2,590,230 parameters before fusion |
| Pretrained source | [Official yolo11n.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt) |
| Pretrained SHA256 | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` |
| Python / system | CPython 3.12.3; Ubuntu 24.04, WSL 2 |
| GPU / Windows host driver | RTX 4070 Ti SUPER, 16,376 MiB; 591.86 |
| PyTorch / torchvision | 2.13.0+cu126 / 0.28.0+cu126 |
| Ultralytics | 8.4.163, pinned training and evaluation implementation |
| Environment | Separate `.venv-ml`, created with the existing uv 0.12.17 |
| Other dependencies | Complete tested pins in [requirements-ml.txt](../requirements-ml.txt) |

The compatible pair and CUDA wheel source follow the [official PyTorch version table](https://pytorch.org/get-started/previous-versions/). Torch/torchvision come from `https://download.pytorch.org/whl/cu126`; Ultralytics and its additional dependencies come from `https://pypi.org/simple`. No Linux NVIDIA driver, Windows driver replacement, system CUDA Toolkit or new environment manager was installed. The `cuda-toolkit` Python metapackage is a dependency of these prebuilt wheels, not a system toolkit installation.

CUDA allocation, YOLO11 forward/backward, finite gradients and an optimizer update were checked on a disposable copy of the selected model. FP32/AMP prediction relative RMSE must be <=0.02. This bypasses only the library's built-in AMP probe, which would download another architecture, and does not alter detector/loss code. A two-epoch smoke experiment on fixed 32/16 train/validation subsets verifies finite losses, updated parameters, save/reload and original-pixel prediction serialization. The main experiment starts again from the original pretrained file.

## Frozen data and evaluation policy

- Official PhenoBench v1.1.0 train/validation membership: 1,407 / 772 images, all 1024 x 1024.
- All 20,016 / 10,408 annotation-valid appearances retained, including 3,566 / 1,840 partial objects.
- Adapter crop=1 -> detector crop=0; weed=2 -> weed=1. Background is excluded as a class.
- Existing half-open visible extents become normalized YOLO xywh, including positive one-pixel boxes.
- Raw IDs, visibility and partial flags stay in local provenance. They are never input features.
- The exporter verifies source RGB hashes, pairings, bounds, counts and split membership against the existing audit. It fails on invalid annotations and float32 label duplicates. Native dataset labels are compared back to the export.
- Project **all-annotated-objects** AP: native Ultralytics evaluator, IoU 0.50:0.05:0.95, plus AP50/AP75. No official PhenoBench ignore filter or leaderboard equivalence is claimed.
- Validation selects maximum mAP50-95. Installed fitness is exactly that metric; native best.pt is overwritten on equal fitness.
- Standalone final evaluation: imgsz=1024, confidence floor=0.001, class-aware NMS IoU=0.7, max_det=300, FP32, no random augmentation. Maximum annotated object count is 35 train / 32 validation.
- Training-time validation uses the native trainer's AMP/FP16 path. Final reported scores come from a fresh process reloading the selected checkpoint with FP32.
- Precision/recall are not reported as fixed-threshold measures: the library's printed summaries select a smoothed max-F1 operating point at IoU 0.5.
- No fixed AP target, architecture comparison or hyperparameter search.

This protocol was fixed before the main experiment. The accepted decision is a local educational portfolio baseline with attribution. The website/archive CC BY-SA 4.0 versus CC BY-NC-SA 4.0 discrepancy and 537 crop IDs shared between train and validation remain unresolved facts. Validation is used for checkpoint selection and is not independent testing on unseen fields. See [DATA](DATA.md).

## Training configuration and artifacts

[configs/yolo11n-baseline.yaml](../configs/yolo11n-baseline.yaml) fixes imgsz=1024, batch=8, device=0, AMP, workers=4, seed=42, 50 epochs, patience=10, AdamW lr0=0.001, weight_decay=0.0005 and warmup_epochs=3. These are starting settings, not optimality claims.

Only HSV gains h=0.01, s=0.2, v=0.2 and horizontal/vertical flips at probability 0.5 are used. The small dataset/trainer hook selects native LetterBox, HSV, flips and Format. RandomPerspective is absent because identity geometry still filters tiny boxes in the native pipeline. No mosaic, mixup, copy-paste, cutmix, crop, rotation, translation, perspective or random scaling is active. Validation uses LetterBox and Format only.

Automatic library details are recorded in effective configuration and epoch manifests: the detection head changes from 80 to 2 classes, 448/499 pretrained tensors transfer, DFL convolution is frozen, nominal batch nbs=64 yields gradient accumulation up to 8, and AdamW weight decay applies only to the weight group. Native defaults include momentum/beta1=0.937 and warmup_bias_lr=0.1; all effective values are saved. Main warmup spans the first three epochs. Smoke warmup is capped by the library to one epoch. CUDA channels-last layout is enabled automatically; compile and distributed training are disabled. Validation loader uses 2x training batch and worker count internally.

The epoch limit stays 50, with native time=null. The separate approximate 10,800-second budget requests a graceful stop; the maintained runner stops after completing an epoch and saving its checkpoint. This can add one epoch and final validation to the approximate limit. SIGINT/SIGTERM request the same graceful boundary. No automatic second experiment or continuation is launched.

Native best.pt/last.pt are kept locally. resume.pt copies the final unstripped optimizer/scaler checkpoint before native final validation strips best/last. Abrupt interruption leaves the last successfully saved native checkpoint; this is not a promise to replay the exact interrupted batch. There is no custom resume CLI.

Local evidence includes effective configuration, all package versions/licenses, source revision and per-file hashes, pretrained/trained weight hashes, per-epoch losses/metrics, native curves, CUDA checks and full logs. Training sources are fingerprinted when the experiment starts.

## Measured result

One main experiment completed **31 epochs**, stopping normally after ten epochs without improvement. The selected checkpoint is from **epoch 21**, with training-time selection fitness 0.63635. The fresh-process FP32 result on all **772 validation images / 10,408 objects** is:

| Class | mAP50-95 | AP50 | AP75 |
|---|---:|---:|---:|
| All | 0.635192 | 0.838123 | 0.674855 |
| Crop | 0.800301 | 0.941166 | 0.856165 |
| Weed | 0.470084 | 0.735080 | 0.493546 |

Scores are on a 0..1 scale. They are validation results under the project protocol, not official PhenoBench leaderboard scores or independent unseen-field testing.

Recorded monotonic duration: **1361.9 seconds** for the training loop including epoch validation, and **1487.6 seconds** for the complete main invocation including setup and final native validation. The native wall-clock CSV ends at 1240.32 seconds; these two timing bases differ in this WSL run, so the reported elapsed duration uses the monotonic record. Batch=8 and workers=4 remained unchanged. Main CUDA/AMP prediction relative RMSE was 0.0002235; real backward and parameter update passed.

Checkpoint: `.local/m2.2/main/weights/best.pt`

SHA256: `ea861ff9ba54a768dffd62814944d157e50037a67fd268ea65fe0d7773196fd6`

The same weights produce original-pixel JSON predictions for all 772 validation images. Eight inspected overlays comprise six preselected ordinary examples and two explicitly selected failures. In `05-15_00157_P0030943`, three small annotated crops are matched only by weed predictions at IoU>=0.5. In `06-05_00217_P0038051`, eleven annotated weeds lack a same-class IoU>=0.5 prediction at confidence 0.25. This diagnostic includes poor localization as well as absence. Ordinary examples also show short weed boxes with insufficient overlap, small crop/weed confusion, and redundant or fragmented boxes in crowded/border regions. The weaker weed AP75 (0.493546) is consistent with the observed localization difficulties. No second training campaign was started.

The native loss curves show falling training losses and plateauing validation AP; validation DFL rises later in training. This is an observed learning-curve pattern, not proof of generalization.

**Verification:** 52 offline synthetic Python tests pass, including the unchanged 36 M2.1 tests and a real CPU-only native validator smoke with network requests forbidden. The original M2.1 environment independently passes its 36 tests. Native data loading preserves all 30,424 objects, including 11 one-pixel-area boxes; all widths/heights are positive. Two real-data exports have 2,185 byte-identical substantive files. Debug and Release CTest each pass 40/40 scenarios. Controller, original audit and Windows-source hashes remain unchanged. CUDA/smoke/full-validation checks are separate from these unit/regression checks.


## Reproduction

For a fresh environment, use the exact, source-separated uv installation in [REPRODUCIBILITY](REPRODUCIBILITY.md#existing-python-suites). It was tested from an empty environment in M5. The historical combined-index command could fail under uv's first-index policy and has been replaced by that procedure. For training commands below, set the environment path explicitly or create the pinned environment at `.venv-ml`; the native demo does not require it.

With the existing original package and M2.1 audit, export to a **new** derived directory. The real-data commands used:

```bash
.venv/bin/python tools/yolo_data.py --dataset-root /home/pc/datasets/phenobench-v110/PhenoBench --audit-root .local/m2.1/run-a --output .local/m2.2/dataset
mkdir -p .local/m2.2/weights
curl -fL --retry 2 https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt -o .local/m2.2/weights/yolo11n.pt
sha256sum .local/m2.2/weights/yolo11n.pt
.venv-ml/bin/python -m unittest discover -s tests -p 'test_*.py' -v
.venv-ml/bin/python tools/train_baseline.py --data .local/m2.2/dataset/smoke.yaml --weights .local/m2.2/weights/yolo11n.pt --output .local/m2.2/smoke --smoke
.venv-ml/bin/python tools/evaluate_baseline.py --weights .local/m2.2/smoke/weights/best.pt --derived .local/m2.2/dataset --output .local/m2.2/smoke-evaluation --smoke
.venv-ml/bin/python tools/train_baseline.py --data .local/m2.2/dataset/dataset.yaml --weights .local/m2.2/weights/yolo11n.pt --output .local/m2.2/main
.venv-ml/bin/python tools/evaluate_baseline.py --weights .local/m2.2/main/weights/best.pt --derived .local/m2.2/dataset --output .local/m2.2/evaluation
```

Arguments, not hard-coded home paths, select source/audit/output locations. The commands above show the measured local paths. Do not reuse an existing run's output or duplicate a running process. Predictions/evaluation use a new output directory. Test data is synthetic: normal tests download no weights and require neither GPU nor real PhenoBench data.

Inspection predictions use confidence=0.25, the same 1024 letterbox, class-aware NMS IoU=0.7, max_det=300 and FP32. JSON records original image ID/dimensions and detection class/name/score/xyxy. Ordinary examples are the first two sorted validation IDs per acquisition date, selected before outputs are reviewed. Additional examples maximize observed missed-weed or crop/weed-confusion counts; their post-hoc selection is labelled. The diagnostic uses GT coverage at IoU>=0.5 for selecting examples, not a replacement AP implementation. Local overlays show GT and predictions in separate labelled panels with different colors.

## Third-party notices

- PhenoBench: attribution to the PhenoBench authors and [dataset/paper](https://www.phenobench.org/dataset.html); the unresolved archive/website license discrepancy is retained in [DATA](DATA.md). No dataset, image derivative or checkpoint is distributed in Git or the source archive.
- Ultralytics 8.4.163, official pretrained weights and resulting YOLO models: [AGPL-3.0](https://www.ultralytics.com/license), as declared by the publisher. ultralytics-thop declares AGPL-3.0; ultralytics-platform declares AGPL-3.0-only. Telemetry, sync and hosted integrations are disabled; no cloud account/tracker is used.
- PyTorch/torchvision: upstream BSD terms plus bundled third-party notices. The installed torch wheel declares Apache-2.0, Apache-2.0 WITH LLVM-exception, BSD-2-Clause, BSD-3-Clause, BSL-1.0 and MIT; torchvision metadata declares BSD.
- NVIDIA CUDA/cuDNN and other NVIDIA wheel components retain their separate [NVIDIA terms](https://docs.nvidia.com/cuda/eula/index.html).
- NumPy's installed wheel declares BSD-3-Clause, 0BSD, MIT, Zlib and CC0-1.0; Pillow declares MIT-CMU; OpenCV declares Apache-2.0. Full distribution metadata/license inventory is kept with local run evidence.

The project's own code license is not selected or changed by this milestone. These third-party notices do not establish commercial deployment or redistribution readiness.

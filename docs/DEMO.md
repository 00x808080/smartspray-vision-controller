# Native vision-to-command demo (M4)

`smartspray_vision_demo` runs decode, ONNX inference, weed selection, original-image center mapping, batch planning, virtual execution and static rendering in C++17. It reuses M3 vision, the shared mapper, controller, OpenCV and nlohmann-json. No Python/service/replay/GPU is invoked. Controller-only builds and prior executables remain available.

## Verified invocation and artifacts

Run from the Linux repository root with the existing [M3 dependencies](INFERENCE.md):

```bash
cmake -S . -B build-vision-release -DCMAKE_BUILD_TYPE=Release \
  -DSMARTSPRAY_BUILD_VISION=ON \
  -DONNXRUNTIME_ROOT="$PWD/.local/m3/deps/onnxruntime-linux-x64-1.22.0" \
  -DSMARTSPRAY_REAL_MODEL="$PWD/.local/m3/model/detector.onnx" \
  -DSMARTSPRAY_REAL_IMAGE="$PWD/.local/m2.2/dataset/images/train/05-15_00028_P0030852.png"
cmake --build build-vision-release --parallel 2
ctest --test-dir build-vision-release --output-on-failure --no-tests=error --verbose

./build-vision-release/smartspray_vision_demo \
  --model .local/m3/model/detector.onnx \
  --image .local/m2.2/dataset/images/train/05-15_00028_P0030852.png \
  --config configs/demo.json \
  --output .local/m4/run-a
```

All four CLI options are required and accept portable paths. Output parent must exist; output directory must not exist (including empty directories/symlinks). Repeat into a fresh folder such as `.local/m4/run-b`. During verification `.local/m4` already held the frozen manifest; create your own parent first.

| Required artifact | SHA256 |
|---|---|
| Existing M3 `detector.onnx` | `1897b7c32c91b73665f9faa179739ae85e664530380971f3699c684ef6f82c35` |
| Training image `05-15_00028_P0030852.png` | `43aa736cfa133817c93ea33269e7835a274577ef3a35c863c27546aa9e8bd9ba` |
| Committed `configs/demo.json` bytes | `39256507b0e8cb7c2d8dedb3d8c40e35a623ea6a6185351c3010574f7335328c` |

Image/model binaries are local and absent from the source archive. This is the first pre-existing M2.2 smoke training ID, also among inspected M2.1 previews. Image ID/hash and the M3 profile were frozen before M4 predictions; the sample was not searched for favorable results. It is a demonstration, not evaluation.

## Simulation and inference

| Configuration field | Value |
|---|---|
| `nozzle_pitch_m` / derived `swath_width_m` | 0.25 / 2.0 |
| `speed_mps` / `lookahead_m` | 2.0 / 2.0 |
| `actuator_delay_us` / `pulse_duration_us` | 50,000 / 100,000 |
| `capture_time_us` / `simulated_processing_delay_us` | 1,000,000 / 100,000 |
| Derived `now_us` | 1,100,000 |

The complete M3 profile is preserved. Configuration is validated before inference, including integer ranges and checked capture+processing addition. Wall-clock durations appear only in console metadata; they never select virtual delays or geometry.

Original-image weed bbox centers use `x_m = swath_width_m * u_px / image_width_px` and `forward_m = lookahead_m * (1 - v_px / image_height_px)`. As in M3, positive box extents must lie inside the image. No camera offset, channel-center rounding or target clamping is added. These are illustrative coordinates, not calibration or verified stems.

Inference stays ONNX Runtime 1.22.0 CPU, FP32 `images [1,3,1024,1024]` to `output0 [1,6,21504]`, sequential threads 1/1. M3 preprocessing, class-aware NMS, score `>0.25`, suppression IoU `>0.7`, maximum 300 and ordering remain unchanged. Crop=0, weed=1. [Full contract and budgets](INFERENCE.md).

## Outputs and actual result

| Output | Content |
|---|---|
| `run.json` | Schema, image identity/hash/dimensions, model hash/shapes, runtime/settings, conventions, all prediction/selection/target/plan records, merged IDs, actual executor log and final states |
| `events.csv` | `event_time_us,nozzle_index,command` directly from the executor log |
| `annotated.png` | All predictions, stable IDs, scores/status ledger, weed anchors, eight zones and simulation label |
| `timeline.png` | Eight rows 0–7, merged COMMAND ON intervals, virtual axis, capture/now, contributing IDs where legible and final OFF |

Detection `dN` and selected target `tN` use graph candidate index N, not array position or tracked plant identity. All individual predictions/plans survive merging. Crops are `not_selected`, not controller failures.

Actual result: **17 predictions = 8 crops + 9 weeds; 9 targets accepted, 0 rejected, 8 merged intervals, 16 events**. Channel 0 merges overlapping `t1542` and `t16709` while retaining both pulses. Channels 4/5 have no commands. All eight channels finish **OFF at 1,974,044 us**. No rejection was fabricated.

Final time is max(now, all OFF times), without an added tick. Valid empty/crops-only/all-rejected results succeed with zero events and eight OFF states. Invalid configuration/image/model/output/inference fails explicitly without a successful action report. Caught write failures remove the newly created directory. `run.json` is finalized last as completion marker; a folder left by process termination without it is incomplete. No overwrite/resume option is provided.

## Executed checks

Existing Ubuntu 24.04 WSL environment; no dependency upgrade, re-export, retraining, full audit or 772-image evaluation.

- Controller-only Debug/Release: **1 CTest entry / 40 scenarios** each.
- Vision Debug/Release: **5/5 CTest entries** each: 40 controller scenarios, 15 existing vision scenarios, 6 harness boundaries, 15 new demo scenarios, plus separately executed real-model smoke.
- ML Python: **103/103 tests**, previous 88 + 15 new CLI tests, no skips. New CLI tests also passed separately in Debug/Release.
- Original data environment: **36/36 tests**.
- Host ASan + UBSan + float-cast-overflow: **4/4 offline CTest entries**, no diagnostics. Private-model smoke is separate; third-party binaries are not instrumented.
- Offline tests cover crop exclusion, original centers, every channel/image boundary, IDs/merge attribution, touching/overlapping intervals, no-action/mixed results, deadline equality/one-us miss, int64 limits, invalid input/output, actual runtime inference failure, alias protection, partial-write cleanup, JSON/CSV/executor consistency and final OFF. Native SHA256 has known empty/short/multi-block vectors.
- Original M3 32-image harness output stayed byte-identical; six historical boundary results stayed equal.
- Frozen demo passed M3 A–D budgets with fresh PyTorch/Python ORT/native inference. Python/C++ ORT detections/targets/plans/intervals/events are exact. PyTorch center differences stay within the unchanged pixel budget; plans, channels, integer times, intervals and events remain exact.
- Two fresh native demo processes yielded byte-identical complete JSON/CSV and identical decoded PNG pixels. Both performed inference.
- Both final PNGs were visually checked against records: all IDs/statuses, zones, eight rows, attribution, capture/now and OFF label.
- One read-only review found acceptance of trailing garbage after config JSON. Strict whole-document parsing and a four-case CLI regression fixed it before final verification.

Executed offline CLI checks (tiny synthetic ONNX fixtures, no private data/model):

```bash
SMARTSPRAY_INFER="$PWD/build-vision-release/smartspray_infer" \
SMARTSPRAY_VISION_DEMO="$PWD/build-vision-release/smartspray_vision_demo" \
.venv-ml/bin/python -m unittest discover -s tests -p 'test_*.py' -v

.venv/bin/python -m unittest discover -s tests -p test_phenobench_data.py -v
```

Debug uses its corresponding executable directory. Without real-model/image CMake arguments, the separate real-model test skips explicitly; missing executable environment variables likewise skip Python compiled-CLI tests. Skips are not passes. Exact commands/exit codes, manifests, logs and review evidence remain local.

## Limits

Predictions are not ground truth. Missed weeds, misclassification and imperfect boxes remain possible; excluding predicted crops does not establish avoidance of crop damage. This training sample does not establish independent field generalization, physical spraying, real-time performance, calibration, safety or chemical savings.

Original image/model files are unchanged. Dataset/model binaries, visual derivatives and session records stay outside Git. Existing PhenoBench license discrepancy/split overlap and separate model/runtime/code rights remain in [DATA](DATA.md), [BASELINE](BASELINE.md) and [INFERENCE](INFERENCE.md). Local presentation is not publication authorization.

Stop after M4. M5 clean reproduction/final packaging remains separate.

# SmartSpray Vision Controller

An educational vision-to-actuation portfolio demo: one image becomes crop/weed detections, selected weed centers, and commands for eight **virtual** nozzle channels.

The native C++17 pipeline decodes the image, runs YOLO11n through ONNX Runtime CPU, maps predicted weed box centers into illustrative coordinates, plans and merges pulses, executes the complete schedule in virtual time, and renders the result. It writes `annotated.png`, `timeline.png`, `run.json` and `events.csv`. No Python, PyTorch, GPU or service is used by the native executable.

The frozen **training-image showcase** gives 17 predictions: 8 crops not selected and 9 accepted weeds, with 0 rejections, 8 merged intervals and 16 events. All eight channels finish OFF at 1,974,044 us. This demonstrates simulated commands, not detector evaluation or physical spraying.

## Choose a reproduction level

| Level | What it needs |
|---|---|
| A. Controller and offline tests | Controller: C++ toolchain only. Full offline regression suite: declared native libraries and pinned Python test dependencies; no private model or full dataset |
| B. Native single-image demo | Declared ONNX model, one image, OpenCV and ONNX Runtime CPU; no Python/PyTorch at runtime |
| C. Training/evaluation | PhenoBench dataset, audit/derived data and pinned ML stack; existing procedures and evidence, not rerun in M5 |

A source-only clone contains **neither trained weights nor dataset images**. The owner supplies the existing verified ONNX file and showcase image as separate local inputs. Their hashes and the retained training/export route are in [reproduction](docs/REPRODUCIBILITY.md); bit-identical retraining is not promised.

## Quickstart

Verified on the existing Ubuntu 24.04 / WSL 2 host, GCC 13.3.0, CMake/CTest 3.28.3 and Unix Makefiles. Start from the repository root. Controller-only:

```bash
cmake -S . -B build-release -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --parallel 2
ctest --test-dir build-release --output-on-failure --no-tests=error --verbose
./build-release/smartspray_demo
```

For the real-image demo, first provision the pinned native dependencies and verified model/image using [the reproduction instructions](docs/REPRODUCIBILITY.md). Set absolute paths to your separate inputs, extracted ORT release and a **new** output directory:

```bash
export ORT_ROOT=/absolute/path/to/onnxruntime-linux-x64-1.22.0
export MODEL=/absolute/path/to/inputs/detector.onnx
export IMAGE=/absolute/path/to/inputs/05-15_00028_P0030852.png
export OUTPUT=/absolute/path/to/outputs/run-a

cmake -S . -B build-vision-release -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release \
  -DSMARTSPRAY_BUILD_VISION=ON -DONNXRUNTIME_ROOT="$ORT_ROOT" \
  -DSMARTSPRAY_REAL_MODEL="$MODEL" -DSMARTSPRAY_REAL_IMAGE="$IMAGE"
cmake --build build-vision-release --parallel 2
ctest --test-dir build-vision-release --output-on-failure --no-tests=error --verbose
mkdir -p "$(dirname "$OUTPUT")"
./build-vision-release/smartspray_vision_demo \
  --model "$MODEL" --image "$IMAGE" --config configs/demo.json --output "$OUTPUT"
```

The output directory itself must not exist. Debug uses a new build directory and `-DCMAKE_BUILD_TYPE=Debug`. The model/image CMake arguments enable the separate real-model smoke; omitting them causes an explicit skip, not a pass.

## Evidence and documentation

- [Reproduction and M5 handoff](docs/REPRODUCIBILITY.md): versions, external hashes, setup, tests and limits.
- [Demo](docs/DEMO.md): configuration, output schema, stable IDs and simulated timing.
- [Inference](docs/INFERENCE.md): fixed ONNX contract, numerical budgets, consistency and measured CPU timing.
- [Baseline](docs/BASELINE.md): training, validation results and failure examples.
- [Data](docs/DATA.md): PhenoBench provenance, annotation policy, license discrepancy and split overlap.
- [Project contract](docs/PROJECT.md) and [controller notes](docs/LEARNING.md): units, boundaries, errors and virtual execution.

Historical full validation mAP50-95: PyTorch **0.635192**, ONNX **0.635255**, under the project's all-annotated-objects protocol. The fixed M3 comparison found exact Python/C++ ONNX outputs and identical commands for 173 real weed targets. These are separate detector-quality and numerical-consistency measurements; M5 checks fresh-checkout reproduction.

## Limits and rights

One image and one simulated pass only: no video, tracking, camera calibration, hardware or real-time control. Predicted box centers are not verified stems. Excluding predicted crops does not establish crop-damage prevention, chemical savings or field safety. No employer approval or independent-field generalization is claimed.

PhenoBench license notices conflict and 537 crop IDs overlap train/validation; the accepted scenario is local educational use with attribution. Ultralytics code/weights carry AGPL-3.0 terms; dependencies retain separate upstream notices. The project's own code license remains unselected. Dataset/model binaries and presentation images stay outside Git; public distribution is a separate decision.

M5 completion is recorded in the reproduction document. Teaching remains deferred; no further implementation milestone is authorized.

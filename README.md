# SmartSpray Vision Controller

A portfolio project for Industrial AI / Robotics: detect plants in an image, map selected targets to simulation coordinates, and schedule commands for eight virtual spray nozzles.

**Implemented:** M1 (M1.1 and M1.2), a C++17 controller that plans one complete synthetic batch before deterministic virtual execution. Each target retains its original pulse or rejection. Overlapping and touching intervals merge within each channel, retaining their source IDs. A complete event schedule drives eight command states and a separate executed-event log.

The native end-to-end demo processes one image once and simulates one pass over its targets. Python crop/weed inference is implemented in M2.2. Native ONNX CPU inference and a same-controller consistency harness are implemented in M3. The controller has no dependency on a neural network, dataset, Python, or GPU.

**M2 implemented:** the PhenoBench audit/annotation adapter and one locally trained YOLO11n crop/weed baseline. The baseline uses all valid annotated appearances, including partial plants, and preserves official train/validation membership. It is an educational portfolio experiment with attribution; conflicting dataset license notices and crop-ID overlap remain documented limitations. Fresh-process validation mAP50-95 is **0.6352** (crop **0.8003**, weed **0.4701**). See [model, results and reproduction](docs/BASELINE.md) and [data audit](docs/DATA.md).

**M3 implemented:** fixed FP32 ONNX export and native C++ image inference. On 32 fixed real images plus 6 numerical fixtures, all parity budgets pass; Python/C++ ONNX outputs are bit-exact. All 173 real weed targets produce identical virtual commands. Full validation mAP50-95 is **0.635255** (delta **+0.000063**). See [inference contract, measurements and commands](docs/INFERENCE.md).

**M4 implemented:** `smartspray_vision_demo` composes native inference, the shared M3 mapper, batch planner and `VirtualExecutor`, and writes JSON/CSV/PNG outputs. The frozen training-image demonstration produced **17 predictions, 9 selected weed targets, 0 rejections, 8 merged command intervals and 16 events**, ending with eight channels OFF. Two fresh processes matched numerically and pixel-for-pixel. See [native demo and reproduction](docs/DEMO.md). This is simulation, not physical spraying or evaluation.

## Build, test, and run

Verified on Ubuntu 24.04 under WSL 2 with G++ 13.3.0, CMake/CTest 3.28.3, C++17, and the Unix Makefiles generator. Run from the project root:

```bash
cmake -S . -B build-debug -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug --parallel 2
ctest --test-dir build-debug --output-on-failure --no-tests=error --verbose
./build-debug/smartspray_demo
./build-debug/smartspray_demo

cmake -S . -B build-release -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --parallel 2
ctest --test-dir build-release --output-on-failure --no-tests=error --verbose
./build-release/smartspray_demo
./build-release/smartspray_demo
```

Both configurations built and passed **1 registered CTest test containing 40 named scenarios**. Checks remain active with `NDEBUG`; the executable returns a nonzero exit code on failure. The original 20 M1.1 scenarios are unchanged. Twenty M1.2 scenarios cover shared failures, duplicate IDs, exact merge boundaries, channel ties, time limits, input preservation, and all 120 permutations of a five-target input. UBSan/float-cast-overflow also passed all 40 scenarios.

Repeated demo runs produced identical output in both configurations:

```text
plans: target_id nozzle_index arrival_time_us on_time_us off_time_us
b 2 1625000 1575000 1700000
a 2 1500000 1450000 1575000
c 3 1500000 1450000 1575000
rejected outside OUT_OF_SWATH
rejected late TOO_LATE
merged: nozzle_index on_time_us off_time_us source_target_ids
2 1450000 1700000 a b
3 1450000 1575000 c
event_time_us nozzle_index command
1450000 2 ON
1450000 3 ON
1575000 3 OFF
1700000 2 OFF
time_us=1800000 channels[0..7]=OFF OFF OFF OFF OFF OFF OFF OFF
```

The demo has fixed inputs and no configurable CLI. The controller library, demo, and test executable use only the C++ standard library.

## Documentation and roadmap

- [Project specification](docs/PROJECT.md): scope, coordinates, timing, errors, and acceptance criteria.
- [Controller notes](docs/LEARNING.md): API behavior and numerical references.
- [Detection baseline](docs/BASELINE.md): frozen label/evaluation policy, trained YOLO11n, measured results, limitations and verified commands.
- [Native demo](docs/DEMO.md): paths/configuration, trace records, static visuals and M4 verification.
- [Native inference](docs/INFERENCE.md): optional C++ build, fixed deployment contract, layered parity, validation regression and CPU timing.
- [Data audit](docs/DATA.md): measured package facts, provenance, label conventions, split limitations, and Python checks.

| Milestone | Result | Current state |
|---|---|---|
| M0 | Scope, assumptions, and acceptance criteria | Documented |
| M1 | C++ controller with synthetic targets and virtual time | M1.1 and M1.2 implemented and tested |
| M2 | Python crop/weed baseline with explicit data-use and split limitations | M2.1 audit/adapter and M2.2 trained baseline complete |
| M3 | ONNX export and C++ inference with consistency checks | Implemented and verified |
| M4 | One end-to-end demo scenario | Native demo implemented and verified |
| M5 | Reproducibility, documentation, and presentation | Not started |

`plan_pulse` remains available for single targets. `plan_batch` returns either a `BatchFailure` (invalid shared configuration/time) or a `BatchPlan` with input-ordered results, merged intervals, and a complete schedule. Every occurrence of a repeated ID is rejected. Valid empty input produces an empty schedule.

## Limitations and open choices

The current program models commands, not fluid behavior. It uses explicit virtual time, no operating-system clock, execution threads, sleeping, or hardware. A `VirtualExecutor` accepts an unchanged successful pulse or a complete valid event schedule. Its explicit preconditions are in [controller.hpp](src/controller.hpp); arbitrary malformed schedules are not validated. Construction owns an immutable schedule and leaves time at zero with all channels OFF until advancement.

The MVP excludes hardware, ROS 2, Kubernetes, cloud infrastructure, Data Loop, a general-purpose framework, and multiple demo scenarios. Video, tracking, and repeated observations of a target are also outside the planned scenario.

The CV task is crop/weed detection with Ultralytics YOLO11n, initialized from official COCO-pretrained weights. PhenoBench v1.1.0 is used locally with attribution under the accepted educational portfolio baseline decision; the license discrepancy remains unresolved. Ultralytics code/weights carry AGPL-3.0 terms. Dataset images, derivatives and checkpoints are excluded from Git. ONNX Runtime 1.22.0 CPU is implemented for M3; see [data and runtime constraints](docs/PROJECT.md#data-runtime). A license for the project's own code has not been selected or changed.

Simulation results do not establish field accuracy, real-time performance, chemical savings, hardware safety, or compliance with an employer's requirements.

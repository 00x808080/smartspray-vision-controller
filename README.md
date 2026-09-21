# SmartSpray Vision Controller

A portfolio project for Industrial AI / Robotics: detect plants in an image, map selected targets to simulation coordinates, and schedule commands for eight virtual spray nozzles.

**Implemented:** M1 (M1.1 and M1.2), a C++17 controller that plans one complete synthetic batch before deterministic virtual execution. Each target retains its original pulse or rejection. Overlapping and touching intervals merge within each channel, retaining their source IDs. A complete event schedule drives eight command states and a separate executed-event log.

The planned end-to-end demo processes one image once and simulates one pass over its targets. Image processing is not implemented. The controller has no dependency on a neural network, dataset, Python, or GPU.

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

| Milestone | Result | Current state |
|---|---|---|
| M0 | Scope, assumptions, and acceptance criteria | Documented |
| M1 | C++ controller with synthetic targets and virtual time | M1.1 and M1.2 implemented and tested |
| M2 | Python CV baseline on a dataset with verified usage rights | Not started |
| M3 | ONNX export and C++ inference with consistency checks | Not started |
| M4 | One end-to-end demo scenario | Not started |
| M5 | Reproducibility, documentation, and presentation | Not started |

`plan_pulse` remains available for single targets. `plan_batch` returns either a `BatchFailure` (invalid shared configuration/time) or a `BatchPlan` with input-ordered results, merged intervals, and a complete schedule. Every occurrence of a repeated ID is rejected. Valid empty input produces an empty schedule.

## Limitations and open choices

The current program models commands, not fluid behavior. It uses explicit virtual time, no operating-system clock, execution threads, sleeping, or hardware. A `VirtualExecutor` accepts an unchanged successful pulse or a complete valid event schedule. Its explicit preconditions are in [controller.hpp](src/controller.hpp); arbitrary malformed schedules are not validated. Construction owns an immutable schedule and leaves time at zero with all channels OFF until advancement.

The MVP excludes hardware, ROS 2, Kubernetes, cloud infrastructure, Data Loop, a general-purpose framework, and multiple demo scenarios. Video, tracking, and repeated observations of a target are also outside the planned scenario.

Dataset, CV task, model, weights, and inference runtime remain undecided. CropAndWeed and ONNX Runtime are candidates only; see [data and runtime constraints](docs/PROJECT.md#data-runtime). No datasets or models have been downloaded. A license for the project's own code has not been selected.

Simulation results do not establish field accuracy, real-time performance, chemical savings, hardware safety, or compliance with an employer's requirements.

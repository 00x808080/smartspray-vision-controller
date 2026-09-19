# SmartSpray Vision Controller

A portfolio project for Industrial AI / Robotics: detect plants in an image, map selected targets to simulation coordinates, and schedule commands for eight virtual spray nozzles.

**Implemented:** M1.1, a C++17 controller for one synthetic target, one nozzle channel, one pulse, and deterministic virtual ON/OFF execution. The planner returns either a complete pulse or a specific rejection. The executor maintains eight command states and an event log.

The planned end-to-end demo processes one image once and simulates one pass over its targets. Image processing and batch planning have not been implemented. The current controller has no dependency on a neural network, dataset, Python, or GPU.

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

Both configurations built and passed **1 registered CTest test containing 20 named scenarios**. Checks remain active with `NDEBUG`; the executable returns a nonzero exit code on failure. Coverage includes the numerical reference, spatial/time boundaries, validation order, numeric ranges, skipped events, repeated time, and rewind without mutation.

Repeated demo runs produced identical output in both configurations:

```text
target_id=synthetic-1
nozzle_index=2
arrival_time_us=1500000
on_time_us=1450000
off_time_us=1550000
event_time_us nozzle_index command
1450000 2 ON
1550000 2 OFF
time_us=1600000 channels[0..7]=OFF OFF OFF OFF OFF OFF OFF OFF
```

The demo has fixed inputs and no configurable CLI. The controller library, demo, and test executable use only the C++ standard library.

## Documentation and roadmap

- [Project specification](docs/PROJECT.md): scope, coordinates, timing, errors, and acceptance criteria.
- [Controller walkthrough](docs/LEARNING.md): real functions and the numerical example.

| Milestone | Result | Current state |
|---|---|---|
| M0 | Scope, assumptions, and acceptance criteria | Documented |
| M1 | C++ controller with synthetic targets and virtual time | M1.1 implemented and tested; M1.2 not started |
| M2 | Python CV baseline on a dataset with verified usage rights | Not started |
| M3 | ONNX export and C++ inference with consistency checks | Not started |
| M4 | One end-to-end demo scenario | Not started |
| M5 | Reproducibility, documentation, and presentation | Not started |

M1.1 covers exactly one target and one pulse. M1.2 would add batch planning, empty input, unique IDs, interval merging, and multiple active channels; none of these are implemented.

## Limitations and open choices

The current program models commands, not fluid behavior. It uses explicit virtual time, no operating-system clock, execution threads, sleeping, or hardware. A `VirtualExecutor` accepts an unchanged successful planner result; it is not an input-validation interface for manually constructed pulses.

The MVP excludes hardware, ROS 2, Kubernetes, cloud infrastructure, Data Loop, a general-purpose framework, and multiple demo scenarios. Video, tracking, and repeated observations of a target are also outside the planned scenario.

Dataset, CV task, model, weights, and inference runtime remain undecided. CropAndWeed and ONNX Runtime are candidates only; see [data and runtime constraints](docs/PROJECT.md#data-runtime). No datasets or models have been downloaded. A license for the project's own code has not been selected.

Simulation results do not establish field accuracy, real-time performance, chemical savings, hardware safety, or compliance with an employer's requirements.

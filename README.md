# SmartSpray Vision Controller

A portfolio project for Industrial AI / Robotics: detect plants in an image, map selected targets to simulation coordinates, and schedule commands for eight virtual spray nozzles.

**Current state:** project scope and controller contracts are documented (M0). The Linux C++17 build environment has been checked separately. No controller application, CV pipeline, or end-to-end demo has been implemented. M1.1 has not started.

The planned demo processes one image once and simulates one pass over its targets. The first controller uses synthetic targets and has no dependency on a neural network, dataset, Python, or GPU.

## Documentation

- [Project specification](docs/PROJECT.md): scope, architecture, coordinates, timing, errors, and acceptance criteria.
- [Controller walkthrough](docs/LEARNING.md): algorithm explanation and a numerical example.

## Roadmap

| Milestone | Planned result | Current state |
|---|---|---|
| M0 | Scope, assumptions, environment assessment, and acceptance criteria | Documented |
| M1 | C++ controller with synthetic targets and virtual time | Not started |
| M2 | Python CV baseline on a dataset with verified usage rights | Not started |
| M3 | ONNX export and C++ inference with consistency checks | Not started |
| M4 | One end-to-end demo scenario | Not started |
| M5 | Reproducibility, documentation, and presentation | Not started |

M1 starts with one target and one pulse (M1.1), followed by batch planning and interval merging (M1.2).

## Environment and verification

The environment check recorded on 2026-09-19 used Ubuntu 24.04.5 LTS on WSL 2, GCC/G++ 13.3.0, CMake/CTest 3.28.3, GNU Make 4.3, and Git 2.43.0. The CMake generator was **Unix Makefiles**.

A separate temporary C++17 project was configured, built, and tested under an ordinary Linux account without sudo. Configure, build, and CTest each returned exit code 0; **1 test ran, 1 passed, 0 failed**. This verifies the toolchain only. It is not a SmartSpray build or an M1.1 test result.

There are no verified application build or run commands yet.

## Limitations and open choices

The MVP is a simulation. It excludes real hardware, ROS 2, Kubernetes, cloud infrastructure, Data Loop, a general-purpose framework, and multiple demo scenarios. Video, tracking, and repeated observations of a target are also outside the planned scenario.

Dataset, CV task, model, weights, and inference runtime remain undecided. CropAndWeed and ONNX Runtime are candidates only; see [data and runtime constraints](docs/PROJECT.md#data-runtime). No datasets or models have been downloaded. A license for the project's own code has not been selected.

Simulation results will not establish field accuracy, real-time performance, chemical savings, hardware safety, or compliance with an employer's requirements.

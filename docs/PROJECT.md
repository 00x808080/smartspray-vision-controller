# Project specification

This document is the M0 design baseline, not a description of implemented behavior. M1.1 has not started. See the [README](../README.md) for the current implementation and environment status.

## Purpose and scope

The objective is one complete vision-to-actuation demo with eight nozzles and an implementation that can be explained and checked. The first controller is independent of ML. There is no verified employer specification for this project.

The scenario processes one image once, then simulates one pass over the resulting targets. M1 uses synthetic coordinates; later CV stages populate the same target contract. Video and overlapping frames are excluded to avoid tracking and duplicate target processing.

Outside the MVP: hardware, ROS 2, Kubernetes, cloud infrastructure, Data Loop, a general-purpose framework, and multiple demo scenarios. No GUI, networking, execution threads, or hardware adapter are planned. An annotated image and a timeline or log of eight channels are sufficient for the demonstration.

## Minimal architecture

These are logical steps, not a requirement for separate classes, processes, or packages.

| Component | Input → output | Purpose / milestone |
|---|---|---|
| CV baseline; later C++ inference | One image → detections with class, score, and a selected point in pixels | Find candidate plants; M2/M3 |
| Target selection and coordinate mapping | Detections + specified geometry → `Target` | Separate plant semantics and pixels from controller logic; M2–M4 |
| Controller planner | `Target` + `Config` + `now_us` → `Pulse` or a rejection reason | Select a channel and command times independently of CV; M1 |
| Virtual execution and event log | Pulses + time advancement → ON/OFF events and eight channel states | Check commands deterministically and present the result; M1/M4 |

M1.1 passes structures directly within one C++ executable. It needs no external interchange format. M2 may use one small JSON file with `schema_version`, units, and targets; define its exact schema when the Python/C++ boundary exists. Network services and a general-purpose serializer are unnecessary.

## Coordinates and units

- The ground is a stationary plane. The machine moves straight at constant `speed_mps > 0`.
- At `capture_time_us`, the origin is at the left edge of the swath on the nozzle bar, viewed in the direction of travel. `x_m` increases to the right; `forward_m` increases forward. Height is not represented.
- `forward_m` is the distance from the nozzle bar **at image capture**, not at processing time. Under this convention, do not add a separate camera offset to that distance.
- Distances use meters (`double`), speed uses meters per second (`double`), and durations and timestamps use integer microseconds (`int64_t`). Field names include units.
- There are eight equal zones. `nozzle_pitch_m = 0.25` and total width `2.0 m` are illustrative values, not a real device specification.
- Channel `i` covers `[i * pitch, (i + 1) * pitch)`, for `i = 0..7`. An internal boundary belongs to the channel on its right; `x_m = 2.0` is outside the swath. After bounds checks, compute the index as `floor(x_m / nozzle_pitch_m)`.

In later image processing, the image origin is at the top left, `u_px` increases rightward, and `v_px` increases downward. Specify the target point and original image dimensions explicitly. The controller does not accept pixel coordinates.

For one image, the proposed **mapping into simulation space** is not camera calibration:

```text
x_m       = width_m * u_px / image_width_px
forward_m = lookahead_m * (1 - v_px / image_height_px)
```

The image domain is `0 <= u_px < image_width_px` and `0 <= v_px < image_height_px`. Choose and disclose `lookahead_m` and image orientation for the single demo. Undo resize/letterbox transforms into original image coordinates first. A bounding-box center is not a verified stem location; the anchor depends on the selected CV task.

## Controller inputs, time, and outputs

| Structure/value | Fields and meaning |
|---|---|
| `Target` | `target_id` (identifier within one pass), `capture_time_us`, `x_m`, `forward_m`; the target has already been selected for treatment |
| `Config` | `nozzle_pitch_m`, `speed_mps`, `actuator_delay_us`, `pulse_duration_us`; channel count is fixed at 8 |
| `now_us` | Time when the target becomes available to the controller; `0 <= capture_time_us <= now_us` |
| `Pulse` | `target_id`, `nozzle_index`, `arrival_time_us`, `on_time_us`, `off_time_us` |
| Rejection | `target_id` and a specific reason; no pulse is created |
| Event log | `event_time_us`, `nozzle_index`, `ON`/`OFF`; channel states are listed in order `0..7` |

Time is monotonic and virtual, measured from a shared run origin. It is neither UTC nor an OS clock. Targets from one frame share `capture_time_us`. The difference `now_us - capture_time_us` already represents the simulated processing delay: do not subtract inference time again. Later, actual measured inference duration is reported as a separate metric.

Calculation on the microsecond grid:

```text
travel_time_us  = ceil(1_000_000 * forward_m / speed_mps)
arrival_time_us = capture_time_us + travel_time_us
on_time_us      = arrival_time_us - actuator_delay_us
off_time_us     = on_time_us + pulse_duration_us
```

Rounding travel time upward is an explicit part of the discrete model. Check finiteness and range before conversion to `int64_t`, and check each operation for overflow. Do not clamp overflow to the maximum value. Compare integer times without an epsilon; test nearby spatial boundaries separately.

If `on_time_us < now_us`, return `TOO_LATE`. Do not backdate the command or replace a missed time with immediate activation. **`on_time_us == now_us` is valid.** Frame age alone is not a rejection reason if the command is still in the future.

## Execution assumptions

- All eight command states start OFF. Nozzles are independent; aggregate flow limits are not modeled.
- Command state ON occupies the half-open interval `[on_time_us, off_time_us)`. At `off_time_us`, the state is already OFF.
- With the same fixed delay for switching on and off, the assumed effective interval is shifted by `actuator_delay_us`. The target reaches the bar at the **start** of the effective pulse, not its midpoint. This is a demo convention; spray width, fluid behavior, and physical plant coverage are not modeled.
- `advance_to(t)` applies every unexecuted event with time `<= t` in chronological order and preserves original timestamps in the log. Advancing directly past OFF still emits both events. Repeating a time emits no duplicate events; moving backward returns `CLOCK_REWIND` without changing state.
- M1.1 accepts exactly one target and one pulse. Before handling multiple targets in M1.2, merge both overlapping **and touching** intervals on the same channel; otherwise an earlier OFF may truncate a longer pulse.
- In M1.2, plan the whole batch before execution. `target_id` values are unique within a pass; reject duplicates. A merged interval retains its source IDs. Order simultaneous events on different channels by channel index. All channels are OFF after a complete pass.

## Error handling

| Condition | Behavior |
|---|---|
| Non-finite configuration geometry, nonpositive pitch/speed/pulse, or negative delay | `INVALID_CONFIG`; the run does not start |
| NaN/Infinity in target coordinates | `INVALID_TARGET`; no events |
| `x_m < 0` or `x_m >= 8 * pitch` | `OUT_OF_SWATH`; do not clamp to an edge channel |
| `forward_m < 0` | `BEHIND_BAR`; no events |
| Negative timestamp or a frame from the future | `INVALID_TIMESTAMP`; no events |
| A calculated time or index cannot be represented correctly | `TIME_OUT_OF_RANGE` for time, `INVALID_TARGET` for an index; no events |
| The ON command is already late | `TOO_LATE`; no events |
| Virtual time moves backward | `CLOCK_REWIND`; preserve state and log |

Validation order: configuration → timestamps → coordinate finiteness → swath bounds → forward distance → arithmetic representability → lateness.

In a future batch, reject an invalid target independently without removing valid commands for other targets. An empty batch is valid and produces no events. At later stages, an image/model loading failure stops the pass before planning; this differs from a valid result with zero detections.

## M1.1: one target, one pulse

The first implementation slice is a minimal C++17 executable that calculates one pulse and executes it virtually. No neural network, GPU, or Python is needed. A Linux toolchain smoke test establishes environment readiness only; the first actual application build and tests belong to M1.1.

Planned files; none are implemented yet:

| File | Purpose |
|---|---|
| `CMakeLists.txt` | One controller library target, a demo executable, a test executable, and CTest |
| `src/controller.hpp` | Simple structures, result type, and function declarations |
| `src/controller.cpp` | Validation, pulse calculation, and minimal virtual execution |
| `src/main.cpp` | One fixed synthetic target and printed output |
| `tests/controller_tests.cpp` | Numerical reference, boundaries, rejections, and time-dependent states |

No external test framework, JSON parser, CLI configurator, or separate clock/executor hierarchy is needed. The test executable returns a nonzero code on failure; checks must remain active under `NDEBUG`.

Reference input: `pitch = 0.25 m`, `x = 0.625 m`, `forward = 1.0 m`, `speed = 2.0 m/s`, `capture = 1_000_000 us`, `now = 1_100_000 us`, `delay = 50_000 us`, `pulse = 100_000 us`. Expected: channel `2`, arrival `1_500_000 us`, ON `1_450_000 us`, OFF `1_550_000 us`. See the [walkthrough](LEARNING.md).

M1.1 acceptance criteria (**not yet tested**):

| Check | Expected result |
|---|---|
| Build and run demo/CTest in the selected environment | Recorded commands and versions; successful exit codes |
| Reference input above | Exactly ON/OFF for channel 2 at the specified timestamps; identical output on a repeated run |
| `x = 0`, `0.25`, `0.625`, `1.999`, `2.0`, negative x | Channels `0`, `1`, `2`, `7`, followed by two `OUT_OF_SWATH` rejections |
| Adjacent representable `double` values to the left/right of `0.25` | Channels `0`/`1`; no artificial epsilon |
| `forward = 1`, `speed = 3` | `travel_time_us = 333_334`, verifying upward rounding |
| `forward = 0`, `capture = now`, `delay = 0` | ON now is valid; with positive delay, return `TOO_LATE` |
| ON equals now / precedes now by 1 us | Accept / `TOO_LATE`; no event on rejection |
| Old capture time, but ON still in the future | Accept; do not introduce an unspecified timeout |
| Zero/negative speed/pitch/pulse, negative delay; NaN/Infinity in double fields | Appropriate error, without division or events |
| Negative forward distance; invalid timestamps | `BEHIND_BAR` / `INVALID_TIMESTAMP` |
| Excessive travel time, arrival overflow, or OFF overflow | `TIME_OUT_OF_RANGE`; no undefined behavior or partial plan |
| Times `1_449_999`, `1_450_000`, `1_549_999`, `1_550_000 us` | All OFF; only channel 2 ON; channel 2 still ON; all OFF |
| Advance directly to `1_600_000 us` | Both events logged with their original timestamps |
| Repeat a time / rewind time | No duplicate events / `CLOCK_REWIND` without mutation |

The remaining M1 slice, M1.2, covers one batch, empty input, ID uniqueness, interval merging, and independent channels. Required same-channel merge cases, in microseconds:

- `[100,200)` and `[150,250)` → `[100,250)`.
- `[100,200)` and `[200,300)` → `[100,300)`.
- `[100,200)` and `[201,300)` remain separate intervals.

Completing M1 requires these checks; M1.1 alone is not the whole milestone.

## Later milestones and acceptance criteria

| Milestone | Completion condition |
|---|---|
| M2 | Verify rights for one dataset/version and its artifacts; select task, classes, a split without leakage, metric, and one baseline. Measure held-out performance; retain errors and provenance. Set the numerical target after the data audit and before evaluating the final model |
| M3 | Select one runtime/provider and pin versions and ONNX opset/IR. Use identical preprocessing/postprocessing. Compare input tensors, raw outputs, and targets on fixed examples with tolerances specified in advance; show discrepancies that change channels or commands |
| M4 | One example permitted for demonstration runs from image to the eight-channel log. Expose the illustrative geometry, rejected targets, timing assumptions, and command reasons; all channels finish OFF |
| M5 | Reproduce the result from a clean environment using verified instructions, declared versions/checksums, tests, sources/licenses, one demo artifact, and explicit limitations |

<a id="data-runtime"></a>

## Data and runtime: unresolved choices

The [official CropAndWeed license](https://github.com/cropandweed/cropandweed-dataset/blob/main/LICENCE) was reviewed on 2026-09-19. It permits noncommercial use, prohibits commercial use of the data and derivatives, and prohibits redistribution of the dataset or modified versions. It provides limited redistribution permission for abstract derivatives, such as models from which the data cannot be reconstructed. These terms do not establish permission for the intended career portfolio use or image publication. Before selecting the dataset, verify the specific use and display rights; if unclear, use a source with suitable rights or obtain clarification from the rights holder. The dataset has not been selected or downloaded.

ONNX Runtime is only a candidate. The main repository's [MIT license](https://github.com/microsoft/onnxruntime/blob/main/LICENSE) does not determine rights to data, weights, or training code. Its [compatibility documentation](https://onnxruntime.ai/docs/reference/compatibility.html) describes runtime/environment and ONNX opset/IR compatibility; no particular combination has been validated for this project. The proposal is to check C++ inference on CPU first and consider acceleration only if evidence justifies it.

Before M2, choose: all plants or weeds; detection or segmentation; the target anchor; dataset/version/license and a permitted example; model and weight provenance; evaluation without leakage between related frames. Before M3, choose runtime/provider, exact versions, preprocessing, and tolerances. These choices remain open.

Simulation does not establish field accuracy, real-time performance, chemical savings, hardware safety, or compliance with employer requirements. CV quality, controller arithmetic, and Python/C++ consistency are evaluated separately.

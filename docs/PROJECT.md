# Project specification

This document retains the M0 coordinate/time contract and records the explicit M1.2 batch clarifications. M1.1/M1.2, the M2.1/M2.2 Python data/detection baseline, and M3 native ONNX inference are implemented and tested. M4 and later milestones have not started. See the [README](../README.md) for implemented capabilities and verified build/run commands.

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
| Controller planner | Targets + one `Config` + one `now_us` → original results, merged intervals, and schedule, or a shared failure | Select a channel and command times independently of CV; M1 |
| Virtual execution and event log | Complete schedule + time advancement → ON/OFF events and eight channel states | Check commands deterministically and present the result; M1/M4 |

M1 passes structures directly within one C++ executable. It needs no external interchange format. M2 may use one small JSON file with `schema_version`, units, and targets; define its exact schema when the Python/C++ boundary exists. Network services and a general-purpose serializer are unnecessary.

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
| `Rejection` | `target_id` and a specific reason; no pulse is created |
| `MergedInterval` | `nozzle_index`, `on_time_us`, `off_time_us`, sorted unique `source_target_ids`; no synthetic target ID or arrival time |
| `BatchPlan` | Input-ordered `target_results` containing original `Pulse`/`Rejection` records, `merged_intervals`, and the complete `schedule` |
| `BatchFailure` | Shared `INVALID_CONFIG` or `INVALID_TIMESTAMP`; a separate alternative with no executable schedule |
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
- `plan_pulse` retains single-target behavior. Batch planning merges both overlapping **and touching** intervals on the same channel before constructing events; an earlier OFF cannot truncate a longer merged interval.
- Plan the whole batch before execution. Every occurrence of a repeated `target_id` is rejected. A merged interval retains all contributing IDs uniquely in lexicographic order. Order events by `(event_time_us, nozzle_index)`. All channels are OFF after a complete pass.
- `VirtualExecutor` owns one immutable schedule, a `next_event_` cursor, and a separate executed-event log. Its single-`Pulse` constructor delegates to the same schedule execution path.
- Executor precondition: an unchanged successful `Pulse`, an unchanged successful `BatchPlan::schedule`, or an equivalent complete valid schedule. Events have nonnegative times, channels in `[0,8)`, and are ordered by time/channel. Per channel they strictly alternate ON/OFF, begin ON, end OFF, and describe positive half-open intervals separated by positive gaps. Empty schedules are valid. Arbitrary malformed schedules are not validated. Construction starts at time zero, all OFF, with no events emitted.

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
| Repeated ID anywhere in the batch | Every occurrence receives `DUPLICATE_TARGET_ID`, before individual validation |
| Virtual time moves backward | `CLOCK_REWIND`; preserve state and log |

Single-target validation order (unchanged): configuration → timestamps → coordinate finiteness → swath bounds → forward distance → arithmetic representability → lateness.

In a batch, reject an invalid target independently without removing valid commands for other targets. Empty input is valid only after shared configuration/time validation succeeds and then produces no events. At later stages, an image/model loading failure stops the pass before planning; this differs from a valid result with zero detections.

## M1.1: one target, one pulse

The first implementation slice is a minimal C++17 executable that calculates one pulse and executes it virtually. No neural network, GPU, or Python is needed. A Linux toolchain smoke test establishes environment readiness only; the first actual application build and tests belong to M1.1.

Implemented M1.1 files:

| File | Purpose |
|---|---|
| `CMakeLists.txt` | One controller library target, a demo executable, a test executable, and CTest |
| `src/controller.hpp` | Simple structures, result type, and function declarations |
| `src/controller.cpp` | Validation, pulse calculation, and minimal virtual execution |
| `src/main.cpp` | One fixed synthetic batch and printed output (updated in M1.2) |
| `tests/controller_tests.cpp` | Numerical reference, boundaries, rejections, and time-dependent states |

No external test framework, JSON parser, CLI configurator, or separate clock/executor hierarchy is needed. The test executable returns a nonzero code on failure; checks must remain active under `NDEBUG`.

Reference input: `pitch = 0.25 m`, `x = 0.625 m`, `forward = 1.0 m`, `speed = 2.0 m/s`, `capture = 1_000_000 us`, `now = 1_100_000 us`, `delay = 50_000 us`, `pulse = 100_000 us`. Expected: channel `2`, arrival `1_500_000 us`, ON `1_450_000 us`, OFF `1_550_000 us`. The reference remains in regression tests; see the [controller notes](LEARNING.md).

M1.1 acceptance criteria (**all original 20 scenarios retained unchanged and passing within the current 40-scenario Debug/Release/UBSan runs; the original single-target demo reference is retained in regression tests**):

| Check | Expected result |
|---|---|
| Build and run demo/CTest in the selected environment | Recorded commands and versions; successful exit codes |
| Reference input above | Exactly ON/OFF for channel 2 at the specified timestamps; retained in `reference_plan` and `batch_singleton` |
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

## M1.2: complete batch and merged schedule

Shared validation happens even for empty input: validate one `Config` first, then the shared `now_us`. Invalid configuration returns batch-level `INVALID_CONFIG`; negative `now_us` returns batch-level `INVALID_TIMESTAMP`. `BatchResult` is a variant of `BatchPlan` and `BatchFailure`, so shared failure has neither target results nor an executable schedule.

After shared validation, count IDs across the entire input before calculating any target. Every occurrence of a repeated ID receives `DUPLICATE_TARGET_ID`, including otherwise invalid members of that group. A first occurrence is never accepted or reserved after successful calculation. Different IDs with identical geometry remain distinct accepted targets. Unique-ID targets use `plan_pulse` with its unchanged validation order and arithmetic.

Keep `target_results` in input order, retaining original accepted pulses and per-target rejections. Copy accepted command intervals, sort by channel/ON/OFF, and merge linearly when the next ON is `<=` the current OFF on the same channel. Extend OFF with `max`; retain even a one-microsecond positive gap. Timestamp comparisons use integers directly, without epsilon, `+1`, or subtraction. Store every contributing ID uniquely in lexicographic order.

`merge_intervals` accepts a copied vector of valid intervals: channel in `[0,8)`, `0 <= ON < OFF`, and a nonempty source-ID list. It returns canonical channel/ON order. `make_schedule` requires valid intervals already merged within each channel and prepares all ON/OFF events before execution. Touching same-channel OFF/ON pairs therefore disappear. Inputs are not mutated, and unique-ID input permutations produce identical merged intervals, schedules, and executed logs.

Verified acceptance criteria: Debug and Release configure/build/demo/CTest succeeded; each configuration ran 1 CTest test containing 40 passing scenarios (20 retained M1.1 + 20 M1.2). Repeated demo output matched an independent literal reference. UBSan/float-cast-overflow passed all 40 scenarios with no diagnostics. A deliberately incorrect local expectation under `NDEBUG` exited 1, confirming checks remain active.

| Criterion | Verified expectation / test |
|---|---|
| Empty input/executor; shared configuration/time precedence | `batch_empty`, `batch_shared_errors`; shared failures contain no schedule |
| Singleton regression and original pulses | `batch_singleton`; original arrival/ON/OFF retained |
| Mixed valid/invalid targets, TOO_LATE, overflow and validation order | `batch_mixed_rejections`, `batch_unique_validation_order`, `batch_time_limits` |
| All repeated IDs rejected; invalid duplicate member; distinct IDs with identical geometry | `batch_duplicate_ids`, `batch_identical_geometry` |
| `[100,200)` + `[150,250)` → `[100,250)` | `merge_overlap` |
| `[100,200)` + `[200,300)` → `[100,300)` | `merge_touch`; no event at the touching boundary |
| `[100,200)` + `[201,300)` remain separate | `merge_one_microsecond_gap`; OFF at 200, ON at 201 |
| Identical, nested, transitive and unsorted intervals; complete sorted unique source IDs | `merge_identical_and_nested`, `merge_transitive_unsorted` |
| Independent channels and simultaneous ON/OFF ordering | `merge_channel_independence`, `schedule_cross_channel_ties` |
| One large advance vs incremental; repeated time; rewind without clock/state/log mutation | `batch_advance_repeat_rewind` |
| Timestamp limits, including OFF at INT64_MAX and a one-microsecond gap near it | `merge_timestamp_limits`, `batch_time_limits` |
| All 120 input permutations; caller input unchanged; planning has no execution side effects | `batch_input_permutations`, `batch_planning_purity` |
| One deterministic batch demo; final eight channels OFF | `batch_demo_reference` and repeated Debug/Release demo runs |

M1 is complete for the specified virtual command model. The M1 controller includes no streaming insertion, cancellation, mutable schedules, tracking, multiple-frame handling, threads, OS clocks, hardware, or CV integration.

## M2.1: data audit and annotation adapter

Implemented the reproducible PhenoBench v1.1.0 audit and ground-truth crop/weed detection adapter; see [DATA](DATA.md) for measured results, sources, checks, and reproduction. It preserves official train/val membership, raw instance IDs, partial/visibility information, and half-open original-pixel boxes. The proposed later weed bbox center is only a simulation anchor, not a verified stem position.

## M2.2: one trained crop/weed baseline

The accepted local educational portfolio baseline uses Ultralytics YOLO11n with official COCO-pretrained weights. It preserves official PhenoBench train/val membership and includes all valid annotated appearances, including partial plants and one-pixel visible extents. Classes are crop=0 and weed=1; IDs remain provenance only. No dataset, image derivative or model binary is committed.

The project all-annotated-objects protocol uses native Ultralytics AP50-95, AP50 and AP75, with the checkpoint selected by validation mAP50-95. It does not reproduce the official PhenoBench evaluator. The license discrepancy and 537 shared crop IDs remain facts; the accepted local-use decision does not establish publication/commercial rights or independent unseen-field testing. No numerical AP target or extra training campaign is imposed.

One main experiment is bounded by 50 epochs, early stopping patience 10 and an approximately three-hour training budget. Only conservative HSV changes and flips are used. A label-preserving dataset/trainer hook avoids native identity-perspective small-box filtering. See [BASELINE](BASELINE.md) for measured settings, metrics, provenance, failures and reproduction.

Predictions contain classes, confidence and original-image pixel boxes. That M2.2 stage did not produce metric-space controller targets, commands or ONNX integration. Its original reports remain intact. M3 is recorded separately below.

## M3: native CPU inference and consistency

Implemented optional C++ image inference using ONNX Runtime 1.22.0 CPU, FP32 batch 1, fixed 1024 NCHW, raw decoded two-class output and deterministic external NMS. Controller-only builds retain no vision dependencies. The [inference contract and results](INFERENCE.md) define preprocessing, error behavior, numerical budgets, actual comparisons, full validation regression and CPU timing.

The frozen 32-real/6-synthetic sample passes all numerical gates. Python/C++ ONNX raw outputs are bit-exact; 173 real weed targets produce identical results through the unchanged batch planner/executor. Synthetic channel, lateness and one-microsecond boundaries expose sensitivity separately. This consistency harness uses the illustrative pixel mapping with width/lookahead 2 m, speed 2 m/s, capture/now 1,000,000/1,100,000 us, delay 50,000 us and duration 100,000 us; it is not camera calibration.

One ONNX evaluation preserves the M2.2 all-annotated-objects protocol on 772 validation images, with mAP50-95 0.635255 versus 0.635192. No retraining or dataset investigation was performed. Stop after M3; M4 presentation and hardware integration remain separate.

## Later milestones and acceptance criteria

| Milestone | Completion condition |
|---|---|
| M2 | One local educational crop/weed baseline under the accepted data-use decision; fixed all-annotated-objects protocol, preserved official split and disclosed overlap; measured validation AP, failure examples, provenance and reproducible tests. No independent-field or official-benchmark claim |
| M3 | Select one runtime/provider and pin versions and ONNX opset/IR. Use identical preprocessing/postprocessing. Compare input tensors, raw outputs, and targets on fixed examples with tolerances specified in advance; show discrepancies that change channels or commands |
| M4 | One example permitted for demonstration runs from image to the eight-channel log. Expose the illustrative geometry, rejected targets, timing assumptions, and command reasons; all channels finish OFF |
| M5 | Reproduce the result from a clean environment using verified instructions, declared versions/checksums, tests, sources/licenses, one demo artifact, and explicit limitations |

<a id="data-runtime"></a>

## Data and runtime: unresolved choices

The [official CropAndWeed license](https://github.com/cropandweed/cropandweed-dataset/blob/main/LICENCE) was reviewed on 2026-09-19. It permits noncommercial use, prohibits commercial use of the data and derivatives, and prohibits redistribution of the dataset or modified versions. It provides limited redistribution permission for abstract derivatives, such as models from which the data cannot be reconstructed. These terms do not establish permission for the intended career portfolio use or image publication. Before selecting the dataset, verify the specific use and display rights; if unclear, use a source with suitable rights or obtain clarification from the rights holder. CropAndWeed has not been selected or downloaded. PhenoBench v1.1.0 is the selected local educational baseline dataset; its separate rights conflict remains recorded in [DATA](DATA.md).

M3 validates ONNX Runtime 1.22.0 CPU with opset 18 / IR 8 in Python and native C++. The runtime's [MIT license](https://github.com/microsoft/onnxruntime/blob/v1.22.0/LICENSE) does not determine rights to data, weights, or training code. Its [compatibility documentation](https://onnxruntime.ai/docs/reference/compatibility.html) describes runtime/environment and ONNX opset/IR compatibility. The exact validated stack, dependency sources and limits are recorded in [INFERENCE](INFERENCE.md); accelerator runtimes are outside M3.

M2.1 adopts crop/weed detection and proposes a weed bbox center for later simulation. M2.2 fixes partial treatment, YOLO11n/weight provenance and the limited validation protocol under the accepted local-use decision. Dataset publication rights and image display outside local review remain unresolved. M3 fixes the CPU runtime/provider, exact versions, preprocessing and comparison tolerances in [INFERENCE](INFERENCE.md). The project's own code license remains unselected; this milestone does not invent or change it.

Simulation does not establish field accuracy, real-time performance, chemical savings, hardware safety, or compliance with employer requirements. CV quality, controller arithmetic, and Python/C++ consistency are evaluated separately.

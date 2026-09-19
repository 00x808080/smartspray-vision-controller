# Controller walkthrough

This walkthrough explains the proposed planner and virtual execution model. The numbers are reference expectations, not output from an implemented controller. The full contract is in the [project specification](PROJECT.md).

## From a target to a pulse

The planner answers two questions: which channel covers a target, and when should ON/OFF commands be sent? It does not depend on whether the target came from a synthetic example or a neural network. Geometry and timing can therefore be checked before selecting a dataset.

Consider a 2 m swath divided into eight zones of 0.25 m. Looking forward along the machine's direction of travel, the target is at `x = 0.625 m` from the left edge and `forward = 1.0 m` ahead of the nozzle bar at capture time. Machine speed is `2.0 m/s`.

| Quantity | Value |
|---|---|
| `capture_time_us` | `1_000_000` (1.000 s) |
| `now_us` | `1_100_000` (1.100 s) |
| `actuator_delay_us` | `50_000` (0.050 s) |
| `pulse_duration_us` | `100_000` (0.100 s) |

Calculation:

1. `floor(0.625 / 0.25) = 2`: select the third nozzle, **index 2**, covering `[0.50, 0.75)` m.
2. `1.0 / 2.0 = 0.5 s`: travel time from the captured target position to the nozzle bar.
3. `1.000 + 0.500 = 1.500 s`: expected arrival. Add travel time to capture time because the distance was measured then.
4. `1.500 - 0.050 = 1.450 s`: send ON early enough to account for the channel's fixed delay.
5. `1.450 + 0.100 = 1.550 s`: send OFF after the configured pulse duration.
6. The current time is `1.100 s`; ON is still in the future, so the target is accepted.

Expected command log:

```text
event_time_us  nozzle_index  command
1450000        2             ON
1550000        2             OFF
```

All channels are OFF before 1.450 s. Only channel 2 is ON during `[1.450, 1.550)` s; all channels are OFF from 1.550 s onward.

With the same fixed delay for switching on and off, the assumed effective pulse occupies `[1.500, 1.600)` s. The target arrives at the bar at the start of that interval. The command log does not establish where fluid would land; spray physics is not modeled.

## Absolute time and boundary cases

Do not subtract the processing delay `now - capture = 0.100 s` a second time: the absolute timestamps already account for it.

If the same target becomes available at `now = 1.450001 s`, ON is late by 1 us: return `TOO_LATE` and emit no events. At `now = 1.450000 s`, ON is allowed immediately. No epsilon is used for integer time comparisons.

Travel time is rounded upward to whole microseconds. For `forward = 1 m` and `speed = 3 m/s`, `ceil(1_000_000 / 3) = 333_334 us`.

## Virtual execution

Virtual time is a value advanced explicitly by the test. Advancing directly to 1.600 s must record both events with their original timestamps. No `sleep` is needed, and the test does not depend on machine load. Repeating the same time produces no duplicates; rewinding time is rejected.

`Target` and `Pulse` are simple records, similar to Python dataclasses. The planner receives inputs explicitly and returns either a pulse or a rejection reason. Only event execution and virtual-clock position require state. In C++, check `double` finiteness, time-conversion bounds, and `int64_t` overflow.

Multiple same-channel pulses require interval merging before execution: separate ON/OFF pairs can let an earlier OFF truncate a longer overlapping pulse. M1.2 adds overlapping and touching interval handling; M1.1 covers one target only.

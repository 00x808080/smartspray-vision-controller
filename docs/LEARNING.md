# Controller walkthrough

This walkthrough follows the implemented single-target controller. The complete contract is in the [project specification](PROJECT.md); build/run commands are in the [README](../README.md).

## Inputs and pure planning

[src/main.cpp](../src/main.cpp) supplies one `Target`, one `Config`, and the time when the target becomes available. These simple records are declared in [controller.hpp](../src/controller.hpp).

| Input | Reference value |
|---|---|
| `target_id` | `synthetic-1` |
| `capture_time_us` | `1_000_000` |
| `x_m`, `forward_m` | `0.625 m`, `1.0 m` |
| `nozzle_pitch_m`, `speed_mps` | `0.25 m`, `2.0 m/s` |
| `actuator_delay_us`, `pulse_duration_us` | `50_000 us`, `100_000 us` |
| `now_us` | `1_100_000` |

`plan_pulse` in [controller.cpp](../src/controller.cpp) receives these values explicitly and changes no inputs or execution state. `PlanResult` is a `std::variant<Pulse, Rejection>`: it contains one complete pulse or a rejection with the target ID and reason, never a partial plan. `reason_name` gives the printable rejection name.

Validation follows this order:

1. Configuration: finite positive pitch and speed, finite total swath width, nonnegative delay, positive duration.
2. Timestamps: nonnegative capture/current time, with capture no later than now.
3. Finite target coordinates.
4. Swath bounds: `0 <= x_m < 8 * nozzle_pitch_m`.
5. Nonnegative forward distance.
6. Representable channel index and calculated times.
7. Lateness: reject only when `on_time_us < now_us`.

For the reference, all validation stages succeed:

1. `floor(0.625 / 0.25) = 2`: choose channel 2, covering `[0.50, 0.75)` m.
2. `ceil(1_000_000 * 1.0 / 2.0) = 500_000 us`: travel duration.
3. `1_000_000 + 500_000 = 1_500_000 us`: arrival.
4. `1_500_000 - 50_000 = 1_450_000 us`: ON.
5. `1_450_000 + 100_000 = 1_550_000 us`: OFF.
6. ON is later than `now_us = 1_100_000`, so the result is accepted.

Arrival is based on **capture time** because `forward_m` measures distance from the nozzle bar at capture. Using processing time would add the elapsed `100_000 us` again. There is no separate inference-time subtraction or arbitrary frame-age timeout.

The planner widens the travel calculation to `long double` before multiplication, rounds upward with `ceil`, and checks finiteness and the exclusive limit `2^63` before converting to `int64_t`. That bound avoids a potentially rounded floating representation of `INT64_MAX`. Arrival and OFF additions have explicit overflow guards. ON subtraction is safe because both operands are between zero and `INT64_MAX`. A missed negative ON is rejected as `TOO_LATE`; no epsilon or clamping repairs it.

For `forward_m = 1` and `speed_mps = 3`, travel rounds to `333_334 us`. At `x_m = 0.25`, the right-hand channel is 1; `x_m = 2.0` is outside the swath. Tests include the adjacent representable values around `0.25`.

## Mutable virtual execution

The successful `Pulse` is passed to `VirtualExecutor`. Its constructor expects an unchanged result of `plan_pulse`; arbitrary hand-written pulses are outside that API precondition. The planner does not execute anything, and the constructor does not emit events.

The executor retains only:

- the single pulse;
- virtual `now_us_`, initially zero;
- eight `states_`, initially OFF;
- `next_event_`, identifying the next unexecuted ON or OFF;
- `events_`, the event log.

`advance_to(t)` first rejects `t < now_us_` with `CLOCK_REWIND`, before changing anything. Otherwise it processes each pending event with its original time `<= t`, updates the corresponding command state, appends the event, and finally updates the clock. Accessors expose the clock, states, and log without allowing mutation.

The reference state transitions are:

| Virtual time (us) | Channel 2 | Other seven channels | Logged events |
|---|---|---|---|
| `1_449_999` | OFF | OFF | None |
| `1_450_000` | ON | OFF | ON at `1_450_000` |
| `1_549_999` | ON | OFF | Unchanged |
| `1_550_000` | OFF | OFF | Also OFF at `1_550_000` |

Thus the command interval is half-open: `[1_450_000, 1_550_000)`.

The demo advances directly to `1_600_000`. The loop first processes ON at `1_450_000`, then OFF at `1_550_000`. Both timestamps survive in `event_log()` even though the requested time skipped over the whole pulse. All eight values in `channel_states()` finish OFF. Repeating a time creates no duplicate events; rewinding preserves the clock, states, log, and pending execution position.

With the same fixed delay for switching ON and OFF, the assumed effective interval is `[1_500_000, 1_600_000)`. Arrival is at its start. This shifted interval is a modeling assumption, not a second command interval or a claim about where fluid lands.

## Test evidence and boundary of this slice

[controller_tests.cpp](../tests/controller_tests.cpp) has 20 named scenarios inside one CTest executable. `reference_plan` uses independent literal expectations. `half_open_states`, `skip_events`, `repeat_time`, and `rewind_time` check mutable execution. Other cases check rounding, spatial boundaries, invalid inputs, overflow, validation precedence, and planning independence.

The `require` helper throws on failure; the runner reports the failing scenario and returns a nonzero exit code. It does not use `assert`, so Release builds with `NDEBUG` retain the checks.

Pure calculation ends when `plan_pulse` returns its value. Mutable execution begins in `advance_to`: advancing a clock and recording commands are separate from deciding which commands are valid.

This implementation contains no batches or interval merging. Multiple same-channel pulses would require merging overlapping and touching intervals before execution, which belongs to M1.2. Passing M1.1 tests establishes the specified virtual behavior only; it does not validate an entire M1 controller or a physical sprayer.

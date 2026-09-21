# Controller technical notes

The [project specification](PROJECT.md) defines the contract; [README](../README.md) contains verified build/run commands and the current batch demo output. APIs are declared in [controller.hpp](../src/controller.hpp).

## Planning results

`plan_pulse(Target, Config, now_us)` is unchanged: it returns a complete `Pulse` or a `Rejection`. Validation order remains configuration, timestamps, finite coordinates, swath, forward distance, representability, then lateness. ON equal to now is accepted.

Travel is based on capture time, rounded upward on the microsecond grid. The calculation widens to `long double` before multiplication and checks the exclusive `2^63` bound before integer conversion. Arrival/OFF additions retain explicit overflow guards; no clamping or epsilon is used.

The retained single-target regression uses capture `1000000`, forward `1 m`, speed `2 m/s`, delay `50000`, duration `100000`, and channel 2. It expects arrival `1500000`, ON `1450000`, and OFF `1550000` microseconds.

`plan_batch(targets, config, now_us)` first validates shared configuration, then shared time, even for empty input. `BatchFailure` represents these shared errors separately. Afterward it counts IDs across the whole input and rejects every occurrence of repeated IDs before individual validation. Unique-ID targets use `plan_pulse`. `BatchPlan::target_results` retains original pulses/rejections in input order.

## Merging and scheduling

`MergedInterval` contains channel, ON, OFF, and source IDs. It has no single arrival time or target ID. `merge_intervals` copies valid input, sorts by channel/ON/OFF, merges when next ON is `<=` current OFF, and extends OFF with `max`. It then sorts and deduplicates source IDs. A positive gap of one microsecond remains a gap.

`make_schedule` creates both events for each already-merged interval, then sorts the complete schedule by `(event_time_us, nozzle_index)`. There are no same-channel OFF/ON events at touching boundaries. The merged plan and schedule are independent of input order; per-target results follow input order.

The single demo uses capture `1000000`, speed `2 m/s`, delay `50000`, duration `125000`, and now `1100000`:

| Target | Independent calculation / result |
|---|---|
| a, channel 2, forward 1 m | Travel 500000; command `[1450000,1575000)` |
| b, channel 2, forward 1.25 m | Travel 625000; command `[1575000,1700000)` |
| c, channel 3, forward 1 m | Command `[1450000,1575000)` independently |
| outside, x = 2 m | `OUT_OF_SWATH` |
| late, forward 0 m | ON 950000 precedes now: `TOO_LATE` |

Thus channel 2 retains IDs a,b in `[1450000,1700000)`, while channel 3 retains c in `[1450000,1575000)`. The original accepted a/b pulse records remain available.

## Execution boundary

`VirtualExecutor` owns an immutable schedule, virtual clock initially zero, eight initially OFF states, a `next_event_` cursor, and a separate executed-event log. The single-Pulse constructor delegates to the same schedule constructor. Construction emits no events, including events scheduled at zero.

The executor trusts its documented valid-input precondition: a complete sorted schedule of nonnegative timestamps and valid channels, with positive, separated ON/OFF intervals per channel. An unchanged successful batch schedule satisfies this contract; arbitrary malformed schedules are not validated.

`advance_to(t)` rejects rewind before mutation. Otherwise it applies pending events with timestamp `<= t`, retaining original timestamps in the log. Half-open intervals are already OFF at their end. Repeated times do not repeat events. A large advance and incremental advances produce the same completed log and final all-OFF states.

## Verification and limits

[controller_tests.cpp](../tests/controller_tests.cpp) retains all 20 M1.1 scenarios and adds 20 M1.2 scenarios, including direct interval fixtures, end-to-end target checks, all 120 permutations of one mixed input, numeric limits, and planner/executor independence. The throwing `require` helper stays active under `NDEBUG`. Debug, Release, and UBSan checks passed.

Planning decides valid commands; execution records their application to virtual states. This establishes specified virtual command behavior only. It does not establish physical spray accuracy, real-time guarantees, hardware safety, or user understanding.

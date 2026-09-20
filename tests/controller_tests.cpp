#include "controller.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace smartspray;
constexpr auto max_time = std::numeric_limits<std::int64_t>::max();
constexpr double infinity = std::numeric_limits<double>::infinity();
constexpr double nan = std::numeric_limits<double>::quiet_NaN();

Target reference_target() { return {"reference", 1'000'000, 0.625, 1.0}; }
Config reference_config() { return {0.25, 2.0, 50'000, 100'000}; }

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

Pulse expect_pulse(const PlanResult& result) {
    if (const auto* rejection = std::get_if<Rejection>(&result)) {
        throw std::runtime_error(std::string("unexpected rejection: ") +
                                 reason_name(rejection->reason));
    }
    return std::get<Pulse>(result);
}

void expect_rejection(const Target& target, const Config& config,
                      std::int64_t now_us, RejectionReason expected) {
    const auto result = plan_pulse(target, config, now_us);
    const auto* rejection = std::get_if<Rejection>(&result);
    require(rejection != nullptr, "rejection must contain no pulse");
    require(rejection->target_id == target.target_id, "rejection preserves target ID");
    require(rejection->reason == expected,
            std::string("expected ") + reason_name(expected) +
            ", got " + reason_name(rejection->reason));
}

Pulse reference_pulse() {
    return expect_pulse(plan_pulse(reference_target(), reference_config(), 1'100'000));
}

void expect_states(const VirtualExecutor& executor, bool channel_two_on) {
    for (std::size_t i = 0; i < nozzle_count; ++i) {
        const auto expected = i == 2 && channel_two_on ? Command::ON : Command::OFF;
        require(executor.channel_states()[i] == expected,
                "unexpected state of channel " + std::to_string(i));
    }
}

void expect_event(const Event& event, std::int64_t time_us, Command command) {
    require(event.event_time_us == time_us, "original event timestamp");
    require(event.nozzle_index == 2, "event channel");
    require(event.command == command, "event command");
}

void reference_plan() {
    const auto pulse = reference_pulse();
    require(pulse.target_id == "reference", "pulse preserves target ID");
    require(pulse.nozzle_index == 2, "reference channel");
    require(pulse.arrival_time_us == 1'500'000, "reference arrival");
    require(pulse.on_time_us == 1'450'000, "reference ON");
    require(pulse.off_time_us == 1'550'000, "reference OFF");
}

void swath_examples() {
    auto target = reference_target();
    for (const auto& [x, channel] :
         std::vector<std::pair<double, std::size_t>>{{0.0, 0}, {0.25, 1},
                                                    {0.625, 2}, {1.999, 7}}) {
        target.x_m = x;
        require(expect_pulse(plan_pulse(target, reference_config(), 1'100'000))
                    .nozzle_index == channel, "swath example channel");
    }
    for (const auto x : {2.0, -0.001}) {
        target.x_m = x;
        expect_rejection(target, reference_config(), 1'100'000, RejectionReason::OUT_OF_SWATH);
    }
}

void adjacent_boundary() {
    auto target = reference_target();
    target.x_m = std::nextafter(0.25, -infinity);
    require(expect_pulse(plan_pulse(target, reference_config(), 1'100'000))
                .nozzle_index == 0, "left neighbor of internal boundary");
    target.x_m = std::nextafter(0.25, infinity);
    require(expect_pulse(plan_pulse(target, reference_config(), 1'100'000))
                .nozzle_index == 1, "right neighbor of internal boundary");
    target.x_m = std::nextafter(2.0, 0.0);
    require(expect_pulse(plan_pulse(target, reference_config(), 1'100'000))
                .nozzle_index == 7, "last representable position inside swath");
    target.x_m = std::nextafter(0.0, -infinity);
    expect_rejection(target, reference_config(), 1'100'000, RejectionReason::OUT_OF_SWATH);
}

void ceil_travel() {
    const Target target{"rounding", 0, 0.0, 1.0};
    const Config config{0.25, 3.0, 0, 1};
    const auto pulse = expect_pulse(plan_pulse(target, config, 0));
    require(pulse.arrival_time_us == 333'334, "ceil one third of a second");
    require(pulse.on_time_us == 333'334, "rounded ON");
    require(pulse.off_time_us == 333'335, "one microsecond pulse");

    auto nearby = target;
    auto exact = config;
    exact.speed_mps = 1'000'000.0;
    nearby.forward_m = std::nextafter(1.0, 0.0);
    require(expect_pulse(plan_pulse(nearby, exact, 0)).arrival_time_us == 1,
            "travel just below one microsecond");
    nearby.forward_m = std::nextafter(1.0, infinity);
    require(expect_pulse(plan_pulse(nearby, exact, 0)).arrival_time_us == 2,
            "travel just above one microsecond");
}

void zero_forward() {
    const Target target{"zero", 123, 0.625, 0.0};
    Config config{0.25, 2.0, 0, 10};
    const auto pulse = expect_pulse(plan_pulse(target, config, 123));
    require(pulse.arrival_time_us == 123 && pulse.on_time_us == 123 &&
            pulse.off_time_us == 133, "zero distance and zero delay");
    VirtualExecutor executor{pulse};
    require(executor.advance_to(123) == AdvanceResult::OK, "ON now executes");
    expect_states(executor, true);
    config.actuator_delay_us = 1;
    expect_rejection(target, config, 123, RejectionReason::TOO_LATE);

    const Target origin{"origin", 0, 0.625, 0.0};
    config.actuator_delay_us = 0;
    VirtualExecutor at_origin{expect_pulse(plan_pulse(origin, config, 0))};
    require(at_origin.event_log().empty(), "constructor does not execute events");
    expect_states(at_origin, false);
    require(at_origin.advance_to(0) == AdvanceResult::OK, "event at virtual origin");
    expect_states(at_origin, true);
}

void deadline_equality() {
    const auto pulse = expect_pulse(plan_pulse(reference_target(), reference_config(), 1'450'000));
    require(pulse.on_time_us == 1'450'000, "ON equal to now accepted");
    expect_rejection(reference_target(), reference_config(), 1'450'001,
                     RejectionReason::TOO_LATE);
}

void old_capture() {
    const Target target{"old", 0, 0.625, 10.0};
    const auto pulse = expect_pulse(plan_pulse(target, reference_config(), 4'000'000));
    require(pulse.arrival_time_us == 5'000'000, "arrival is based on capture");
    require(pulse.on_time_us == 4'950'000, "no arbitrary frame age timeout");
}

void invalid_config() {
    const auto target = reference_target();
    for (const auto value : {0.0, -1.0, nan, infinity, -infinity}) {
        auto config = reference_config();
        config.nozzle_pitch_m = value;
        expect_rejection(target, config, 1'100'000, RejectionReason::INVALID_CONFIG);
        config = reference_config();
        config.speed_mps = value;
        expect_rejection(target, config, 1'100'000, RejectionReason::INVALID_CONFIG);
    }
    for (const auto duration : {std::int64_t{0}, std::int64_t{-1},
                                std::numeric_limits<std::int64_t>::min()}) {
        auto config = reference_config();
        config.pulse_duration_us = duration;
        expect_rejection(target, config, 1'100'000, RejectionReason::INVALID_CONFIG);
    }
    auto config = reference_config();
    config.actuator_delay_us = -1;
    expect_rejection(target, config, 1'100'000, RejectionReason::INVALID_CONFIG);
    config = reference_config();
    config.nozzle_pitch_m = std::numeric_limits<double>::max();
    expect_rejection(target, config, 1'100'000, RejectionReason::INVALID_CONFIG);
}

void nonfinite_target() {
    for (const auto value : {nan, infinity, -infinity}) {
        auto target = reference_target();
        target.x_m = value;
        expect_rejection(target, reference_config(), 1'100'000, RejectionReason::INVALID_TARGET);
        target = reference_target();
        target.forward_m = value;
        expect_rejection(target, reference_config(), 1'100'000, RejectionReason::INVALID_TARGET);
    }
}

void behind_bar() {
    auto target = reference_target();
    for (const auto forward : {-1.0, std::nextafter(0.0, -infinity)}) {
        target.forward_m = forward;
        expect_rejection(target, reference_config(), 1'100'000, RejectionReason::BEHIND_BAR);
    }
}

void invalid_timestamps() {
    auto target = reference_target();
    expect_rejection(target, reference_config(), -1, RejectionReason::INVALID_TIMESTAMP);
    target.capture_time_us = -1;
    expect_rejection(target, reference_config(), 1'100'000, RejectionReason::INVALID_TIMESTAMP);
    target.capture_time_us = 1'100'001;
    expect_rejection(target, reference_config(), 1'100'000, RejectionReason::INVALID_TIMESTAMP);
}

void time_range() {
    Target target{"range", 0, 0.625, std::numeric_limits<double>::max()};
    Config config{0.25, 1.0, 0, 1};
    expect_rejection(target, config, 0, RejectionReason::TIME_OUT_OF_RANGE);
    target.forward_m = 1.0;
    config.speed_mps = std::numeric_limits<double>::denorm_min();
    expect_rejection(target, config, 0, RejectionReason::TIME_OUT_OF_RANGE);

    target.forward_m = std::ldexp(1.0, 63);
    config.speed_mps = 1'000'000.0;
    expect_rejection(target, config, 0, RejectionReason::TIME_OUT_OF_RANGE);

    target = {"arrival-overflow", max_time - 1, 0.625, 1.0};
    config.speed_mps = 1.0;
    expect_rejection(target, config, max_time - 1, RejectionReason::TIME_OUT_OF_RANGE);

    target = {"off-overflow", max_time, 0.625, 0.0};
    expect_rejection(target, config, max_time, RejectionReason::TIME_OUT_OF_RANGE);
}

void representable_time_edges() {
    Target target{"last-off", max_time - 1, 0.625, 0.0};
    Config config{0.25, 1.0, 0, 1};
    const auto pulse = expect_pulse(plan_pulse(target, config, max_time - 1));
    require(pulse.on_time_us == max_time - 1 && pulse.off_time_us == max_time,
            "largest representable OFF is valid");
    VirtualExecutor executor{pulse};
    require(executor.advance_to(max_time) == AdvanceResult::OK, "advance to INT64_MAX");
    require(executor.event_log().size() == 2, "both events at end of time range");
    expect_states(executor, false);

    target = {"largest-delay", 0, 0.625, 0.0};
    config.actuator_delay_us = max_time;
    config.pulse_duration_us = max_time;
    expect_rejection(target, config, 0, RejectionReason::TOO_LATE);

    target = {"large-travel", 0, 0.625, std::ldexp(1.0, 43)};
    config = {0.25, 1.0, 0, 1};
    require(expect_pulse(plan_pulse(target, config, 0)).arrival_time_us ==
            8'796'093'022'208'000'000LL, "large representable travel");
}

void half_open_states() {
    VirtualExecutor executor{reference_pulse()};
    require(executor.now_us() == 0 && executor.event_log().empty(), "initial execution state");
    expect_states(executor, false);
    for (const auto& [time, on] :
         std::vector<std::pair<std::int64_t, bool>>{{1'449'999, false}, {1'450'000, true},
                                                   {1'549'999, true}, {1'550'000, false}}) {
        require(executor.advance_to(time) == AdvanceResult::OK, "forward time accepted");
        require(executor.now_us() == time, "clock follows advance");
        expect_states(executor, on);
    }
    require(executor.event_log().size() == 2, "exactly two reference events");
    expect_event(executor.event_log()[0], 1'450'000, Command::ON);
    expect_event(executor.event_log()[1], 1'550'000, Command::OFF);
}

void skip_events() {
    VirtualExecutor executor{reference_pulse()};
    require(executor.advance_to(1'600'000) == AdvanceResult::OK, "skip past OFF");
    require(executor.event_log().size() == 2, "skipped events both retained");
    expect_event(executor.event_log()[0], 1'450'000, Command::ON);
    expect_event(executor.event_log()[1], 1'550'000, Command::OFF);
    expect_states(executor, false);
}

void repeat_time() {
    VirtualExecutor executor{reference_pulse()};
    for (const auto& [time, count] :
         std::vector<std::pair<std::int64_t, std::size_t>>{{0, 0}, {1'450'000, 1},
                                                         {1'550'000, 2}, {1'600'000, 2}}) {
        require(executor.advance_to(time) == AdvanceResult::OK, "first advance");
        require(executor.advance_to(time) == AdvanceResult::OK, "equal time accepted");
        require(executor.event_log().size() == count, "no duplicate events");
    }
}

void rewind_time() {
    VirtualExecutor executor{reference_pulse()};
    require(executor.advance_to(-1) == AdvanceResult::CLOCK_REWIND, "negative rewind");
    require(executor.now_us() == 0 && executor.event_log().empty(), "initial rewind unchanged");
    for (const auto time : {1'449'999, 1'450'000, 1'550'000}) {
        require(executor.advance_to(time) == AdvanceResult::OK, "forward advance");
        const auto states = executor.channel_states();
        const auto events = executor.event_log();
        require(executor.advance_to(time - 1) == AdvanceResult::CLOCK_REWIND,
                "rewind rejected");
        require(executor.now_us() == time && executor.channel_states() == states &&
                executor.event_log().size() == events.size(), "rewind preserves clock/states/log");
        for (std::size_t i = 0; i < events.size(); ++i) {
            const auto& actual = executor.event_log()[i];
            require(actual.event_time_us == events[i].event_time_us &&
                    actual.nozzle_index == events[i].nozzle_index &&
                    actual.command == events[i].command, "rewind preserves event contents");
        }
    }
}

void validation_precedence() {
    auto target = reference_target();
    auto config = reference_config();
    config.speed_mps = 0.0;
    target.capture_time_us = -1;
    target.x_m = nan;
    target.forward_m = -1.0;
    expect_rejection(target, config, -1, RejectionReason::INVALID_CONFIG);
    config = reference_config();
    expect_rejection(target, config, -1, RejectionReason::INVALID_TIMESTAMP);
    target.capture_time_us = 0;
    expect_rejection(target, config, 0, RejectionReason::INVALID_TARGET);
    target.x_m = 2.0;
    expect_rejection(target, config, 0, RejectionReason::OUT_OF_SWATH);
    target.x_m = 0.625;
    expect_rejection(target, config, 0, RejectionReason::BEHIND_BAR);
    // OFF overflow must be reported before lateness.
    target = {"overflow-before-late", max_time - 10, 0.625, 0.0};
    config = {0.25, 1.0, 0, 20};
    expect_rejection(target, config, max_time, RejectionReason::TIME_OUT_OF_RANGE);
}

void finite_geometry_extremes() {
    Target target{"large-finite", 0, 0.625, std::numeric_limits<double>::max()};
    Config config{0.25, std::numeric_limits<double>::max(), 0, 1};
    require(expect_pulse(plan_pulse(target, config, 0)).arrival_time_us == 1'000'000,
            "wide intermediates preserve a representable ratio");
    config = {std::numeric_limits<double>::denorm_min(), 1.0, 0, 1};
    target = {"tiny-pitch", 0, 0.0, 0.0};
    require(expect_pulse(plan_pulse(target, config, 0)).nozzle_index == 0,
            "positive subnormal pitch at x zero");
}

void planner_is_pure() {
    const auto target = reference_target();
    const auto config = reference_config();
    const auto first = plan_pulse(target, config, 1'100'000);
    VirtualExecutor executor{expect_pulse(first)};
    require(executor.advance_to(1'450'000) == AdvanceResult::OK, "existing ON state");
    expect_rejection(target, config, 1'450'001, RejectionReason::TOO_LATE);
    expect_states(executor, true);
    require(executor.now_us() == 1'450'000 && executor.event_log().size() == 1,
            "planning rejection cannot mutate executor");
    const auto repeated = expect_pulse(plan_pulse(target, config, 1'100'000));
    require(repeated.arrival_time_us == 1'500'000 && repeated.on_time_us == 1'450'000 &&
            repeated.off_time_us == 1'550'000, "planning independent of mutable execution");
    require(target.target_id == "reference" && target.capture_time_us == 1'000'000 &&
            target.x_m == 0.625 && target.forward_m == 1.0, "target unchanged");
    require(config.nozzle_pitch_m == 0.25 && config.speed_mps == 2.0 &&
            config.actuator_delay_us == 50'000 && config.pulse_duration_us == 100'000,
            "config unchanged");
}

BatchPlan expect_batch(const BatchResult& result) {
    const auto* batch = std::get_if<BatchPlan>(&result);
    require(batch != nullptr, "expected successful batch");
    return *batch;
}

void expect_batch_failure(const std::vector<Target>& targets, const Config& config,
                          std::int64_t now, RejectionReason reason) {
    const auto result = plan_batch(targets, config, now);
    const auto* failure = std::get_if<BatchFailure>(&result);
    require(failure != nullptr && failure->reason == reason,
            "shared failure has no BatchPlan or executable schedule");
}

void expect_result_rejection(const PlanResult& result, const std::string& id,
                             RejectionReason reason) {
    const auto* rejection = std::get_if<Rejection>(&result);
    require(rejection != nullptr, "target must be rejected");
    require(rejection->target_id == id && rejection->reason == reason,
            "per-target rejection ID/reason");
}

void expect_pulse_record(const Pulse& actual, const Pulse& expected) {
    require(actual.target_id == expected.target_id &&
            actual.nozzle_index == expected.nozzle_index &&
            actual.arrival_time_us == expected.arrival_time_us &&
            actual.on_time_us == expected.on_time_us &&
            actual.off_time_us == expected.off_time_us, "complete original pulse record");
}

void expect_intervals(const std::vector<MergedInterval>& actual,
                      const std::vector<MergedInterval>& expected) {
    require(actual.size() == expected.size(), "merged interval count");
    for (std::size_t i = 0; i < expected.size(); ++i) {
        require(actual[i].nozzle_index == expected[i].nozzle_index &&
                actual[i].on_time_us == expected[i].on_time_us &&
                actual[i].off_time_us == expected[i].off_time_us &&
                actual[i].source_target_ids == expected[i].source_target_ids,
                "merged interval endpoints/channel/source IDs at " + std::to_string(i));
    }
}

void expect_events(const std::vector<Event>& actual, const std::vector<Event>& expected) {
    require(actual.size() == expected.size(), "event count");
    for (std::size_t i = 0; i < expected.size(); ++i) {
        require(actual[i].event_time_us == expected[i].event_time_us &&
                actual[i].nozzle_index == expected[i].nozzle_index &&
                actual[i].command == expected[i].command, "ordered event contents");
    }
}

void expect_on_channels(const VirtualExecutor& executor,
                        const std::vector<std::size_t>& on_channels) {
    for (std::size_t i = 0; i < nozzle_count; ++i) {
        const bool on = std::find(on_channels.begin(), on_channels.end(), i) != on_channels.end();
        require(executor.channel_states()[i] == (on ? Command::ON : Command::OFF),
                "batch channel state " + std::to_string(i));
    }
}

void expect_targets(const std::vector<Target>& actual, const std::vector<Target>& expected) {
    require(actual.size() == expected.size(), "target input size unchanged");
    for (std::size_t i = 0; i < expected.size(); ++i) {
        require(actual[i].target_id == expected[i].target_id &&
                actual[i].capture_time_us == expected[i].capture_time_us &&
                actual[i].x_m == expected[i].x_m &&
                actual[i].forward_m == expected[i].forward_m, "target input unchanged");
    }
}

void batch_empty() {
    const auto batch = expect_batch(plan_batch({}, reference_config(), 123));
    require(batch.target_results.empty() && batch.merged_intervals.empty() &&
            batch.schedule.empty(), "valid empty batch");
    VirtualExecutor executor{batch.schedule};
    require(executor.now_us() == 0 && executor.event_log().empty(), "empty initial state");
    expect_on_channels(executor, {});
    require(executor.advance_to(-1) == AdvanceResult::CLOCK_REWIND, "empty negative rewind");
    require(executor.now_us() == 0, "empty rewind preserves clock");
    require(executor.advance_to(max_time) == AdvanceResult::OK, "empty advance to time limit");
    require(executor.advance_to(max_time) == AdvanceResult::OK, "empty repeat time");
    require(executor.advance_to(0) == AdvanceResult::CLOCK_REWIND, "empty rewind");
    require(executor.now_us() == max_time && executor.event_log().empty(), "empty no events");
    expect_on_channels(executor, {});
    require(expect_batch(plan_batch({}, reference_config(), max_time)).schedule.empty(),
            "empty batch at maximum shared time");
}

void batch_shared_errors() {
    const std::vector<std::vector<Target>> inputs{
        {}, {{"dup", -1, nan, -1.0}, {"dup", 0, 0.0, 100.0}}};
    for (const auto& targets : inputs) {
        for (const auto value : {0.0, -1.0, nan, infinity, -infinity}) {
            auto config = reference_config();
            config.speed_mps = value;
            expect_batch_failure(targets, config, -1, RejectionReason::INVALID_CONFIG);
            config = reference_config();
            config.nozzle_pitch_m = value;
            expect_batch_failure(targets, config, 0, RejectionReason::INVALID_CONFIG);
        }
        auto config = reference_config();
        config.nozzle_pitch_m = std::numeric_limits<double>::max();
        expect_batch_failure(targets, config, 0, RejectionReason::INVALID_CONFIG);
        config = reference_config();
        config.actuator_delay_us = -1;
        expect_batch_failure(targets, config, -1, RejectionReason::INVALID_CONFIG);
        for (const auto duration : {std::int64_t{0}, std::int64_t{-1}}) {
            config = reference_config();
            config.pulse_duration_us = duration;
            expect_batch_failure(targets, config, 0, RejectionReason::INVALID_CONFIG);
        }
        for (const auto now : {std::int64_t{-1}, std::numeric_limits<std::int64_t>::min()}) {
            expect_batch_failure(targets, reference_config(), now,
                                 RejectionReason::INVALID_TIMESTAMP);
        }
    }
}

void batch_singleton() {
    const auto batch = expect_batch(plan_batch({reference_target()}, reference_config(), 1'100'000));
    require(batch.target_results.size() == 1, "singleton result");
    const Pulse expected{"reference", 2, 1'500'000, 1'450'000, 1'550'000};
    expect_pulse_record(expect_pulse(batch.target_results[0]), expected);
    expect_pulse_record(reference_pulse(), expected);
    expect_intervals(batch.merged_intervals, {{2, 1'450'000, 1'550'000, {"reference"}}});
    const std::vector<Event> events{{1'450'000, 2, Command::ON}, {1'550'000, 2, Command::OFF}};
    expect_events(batch.schedule, events);
    VirtualExecutor single{expected};
    VirtualExecutor multiple{batch.schedule};
    for (const auto time : {0, 1'449'999, 1'450'000, 1'549'999, 1'550'000, 1'600'000}) {
        require(single.advance_to(time) == AdvanceResult::OK &&
                multiple.advance_to(time) == AdvanceResult::OK, "both executor constructors");
        require(single.now_us() == multiple.now_us() &&
                single.channel_states() == multiple.channel_states(), "singleton equivalence");
        expect_events(multiple.event_log(), single.event_log());
    }
    expect_events(multiple.event_log(), events);
}

void batch_mixed_rejections() {
    const Config config{0.25, 1'000'000.0, 0, 100};
    const std::vector<Target> targets{
        {"a", 0, 0.625, 100.0}, {"late", 0, 0.0, 0.0},
        {"overflow", 0, 0.0, std::numeric_limits<double>::max()},
        {"outside", 0, 2.0, 100.0}, {"b", 0, 0.875, 200.0},
        {"future", 51, nan, -1.0}};
    const auto batch = expect_batch(plan_batch(targets, config, 50));
    require(batch.target_results.size() == 6, "every input has a result");
    expect_pulse_record(expect_pulse(batch.target_results[0]), {"a", 2, 100, 100, 200});
    expect_result_rejection(batch.target_results[1], "late", RejectionReason::TOO_LATE);
    expect_result_rejection(batch.target_results[2], "overflow", RejectionReason::TIME_OUT_OF_RANGE);
    expect_result_rejection(batch.target_results[3], "outside", RejectionReason::OUT_OF_SWATH);
    expect_pulse_record(expect_pulse(batch.target_results[4]), {"b", 3, 200, 200, 300});
    expect_result_rejection(batch.target_results[5], "future", RejectionReason::INVALID_TIMESTAMP);
    expect_intervals(batch.merged_intervals, {{2, 100, 200, {"a"}}, {3, 200, 300, {"b"}}});
    expect_events(batch.schedule, {{100, 2, Command::ON}, {200, 2, Command::OFF},
                                  {200, 3, Command::ON}, {300, 3, Command::OFF}});
}

void batch_duplicate_ids() {
    const Config config{0.25, 1'000'000.0, 0, 100};
    std::vector<Target> targets{{"dup", -1, nan, -1.0}, {"good", 0, 0.625, 100.0},
                                {"dup", 0, 0.875, 200.0}, {"dup", 0, 0.625, 100.0}};
    for (int reverse = 0; reverse < 2; ++reverse) {
        const auto batch = expect_batch(plan_batch(targets, config, 0));
        require(batch.target_results.size() == 4, "duplicate occurrences retain input positions");
        for (std::size_t i = 0; i < targets.size(); ++i) {
            if (targets[i].target_id == "dup") {
                expect_result_rejection(batch.target_results[i], "dup",
                                        RejectionReason::DUPLICATE_TARGET_ID);
            } else {
                expect_pulse_record(expect_pulse(batch.target_results[i]), {"good", 2, 100, 100, 200});
            }
        }
        expect_intervals(batch.merged_intervals, {{2, 100, 200, {"good"}}});
        expect_events(batch.schedule, {{100, 2, Command::ON}, {200, 2, Command::OFF}});
        std::reverse(targets.begin(), targets.end());
    }
    const auto rejected = expect_batch(plan_batch(
        {{"", 0, 0.0, 100.0}, {"", 0, 0.0, 100.0}}, config, 0));
    require(rejected.target_results.size() == 2 && rejected.merged_intervals.empty() &&
            rejected.schedule.empty(), "all-duplicate batch has no commands");
    for (const auto& result : rejected.target_results) {
        expect_result_rejection(result, "", RejectionReason::DUPLICATE_TARGET_ID);
    }
    require(std::string(reason_name(RejectionReason::DUPLICATE_TARGET_ID)) ==
            "DUPLICATE_TARGET_ID", "duplicate reason name");
}

void batch_identical_geometry() {
    const auto batch = expect_batch(plan_batch(
        {{"z", 0, 0.625, 100.0}, {"a", 0, 0.625, 100.0}},
        {0.25, 1'000'000.0, 0, 100}, 0));
    require(batch.target_results.size() == 2, "distinct IDs accepted");
    expect_pulse_record(expect_pulse(batch.target_results[0]), {"z", 2, 100, 100, 200});
    expect_pulse_record(expect_pulse(batch.target_results[1]), {"a", 2, 100, 100, 200});
    expect_intervals(batch.merged_intervals, {{2, 100, 200, {"a", "z"}}});
    expect_events(batch.schedule, {{100, 2, Command::ON}, {200, 2, Command::OFF}});
}

void batch_unique_validation_order() {
    const auto batch = expect_batch(plan_batch(
        {{"time", -1, nan, -1.0}, {"finite", 0, nan, -1.0},
         {"swath", 0, 2.0, -1.0}, {"behind", 0, 0.0, -1.0}},
        reference_config(), 0));
    require(batch.target_results.size() == 4, "unique invalid results retained");
    expect_result_rejection(batch.target_results[0], "time", RejectionReason::INVALID_TIMESTAMP);
    expect_result_rejection(batch.target_results[1], "finite", RejectionReason::INVALID_TARGET);
    expect_result_rejection(batch.target_results[2], "swath", RejectionReason::OUT_OF_SWATH);
    expect_result_rejection(batch.target_results[3], "behind", RejectionReason::BEHIND_BAR);
    require(batch.merged_intervals.empty() && batch.schedule.empty(), "all-invalid empty schedule");
    const auto late_overflow = expect_batch(plan_batch(
        {{"range", max_time - 10, 0.625, 0.0}}, {0.25, 1.0, 0, 20}, max_time));
    expect_result_rejection(late_overflow.target_results[0], "range", RejectionReason::TIME_OUT_OF_RANGE);
}

void batch_time_limits() {
    const auto batch = expect_batch(plan_batch(
        {{"last", max_time - 1, 0.625, 0.0},
         {"off", max_time - 1, 0.875, 1.0},
         {"arrival", max_time - 1, 0.0, 2.0}},
        {0.25, 1'000'000.0, 0, 1}, max_time - 1));
    require(batch.target_results.size() == 3, "near-limit results");
    expect_pulse_record(expect_pulse(batch.target_results[0]),
                        {"last", 2, max_time - 1, max_time - 1, max_time});
    expect_result_rejection(batch.target_results[1], "off", RejectionReason::TIME_OUT_OF_RANGE);
    expect_result_rejection(batch.target_results[2], "arrival", RejectionReason::TIME_OUT_OF_RANGE);
    const std::vector<Event> events{{max_time - 1, 2, Command::ON}, {max_time, 2, Command::OFF}};
    expect_events(batch.schedule, events);
    VirtualExecutor executor{batch.schedule};
    require(executor.advance_to(max_time - 1) == AdvanceResult::OK, "ON near time limit");
    expect_on_channels(executor, {2});
    require(executor.advance_to(max_time) == AdvanceResult::OK, "OFF at time limit");
    expect_on_channels(executor, {});
    expect_events(executor.event_log(), events);
}

void batch_planning_purity() {
    std::vector<Target> targets{{"z", 0, 0.625, 100.0}, {"bad", 0, 2.0, 0.0},
                                {"a", 0, 0.625, 200.0}};
    const auto original = targets;
    const Config config{0.25, 1'000'000.0, 0, 100};
    VirtualExecutor existing{reference_pulse()};
    require(existing.advance_to(1'450'000) == AdvanceResult::OK, "existing execution underway");
    const auto states = existing.channel_states();
    const auto events = existing.event_log();
    const auto batch = expect_batch(plan_batch(targets, config, 0));
    expect_batch_failure(targets, config, -1, RejectionReason::INVALID_TIMESTAMP);
    expect_targets(targets, original);
    require(config.nozzle_pitch_m == 0.25 && config.speed_mps == 1'000'000.0 &&
            config.actuator_delay_us == 0 && config.pulse_duration_us == 100, "batch config unchanged");
    require(existing.now_us() == 1'450'000 && existing.channel_states() == states,
            "planning has no clock or state side effect");
    expect_events(existing.event_log(), events);
    auto schedule = batch.schedule;
    VirtualExecutor executor{schedule};
    schedule.clear();
    targets.clear();
    require(executor.now_us() == 0 && executor.event_log().empty(), "construction does not execute");
    expect_on_channels(executor, {});
    require(executor.advance_to(300) == AdvanceResult::OK, "executor owns its schedule");
    expect_events(executor.event_log(), {{100, 2, Command::ON}, {300, 2, Command::OFF}});
    expect_on_channels(executor, {});
}


void merge_overlap() {
    expect_intervals(merge_intervals({{2, 100, 200, {"a"}}, {2, 150, 250, {"b"}}}),
                     {{2, 100, 250, {"a", "b"}}});
}

void merge_touch() {
    const auto merged = merge_intervals({{2, 100, 200, {"a"}}, {2, 200, 300, {"b"}}});
    expect_intervals(merged, {{2, 100, 300, {"a", "b"}}});
    expect_events(make_schedule(merged), {{100, 2, Command::ON}, {300, 2, Command::OFF}});
    VirtualExecutor executor{make_schedule(merged)};
    require(executor.advance_to(200) == AdvanceResult::OK, "advance through touching boundary");
    expect_on_channels(executor, {2});
    expect_events(executor.event_log(), {{100, 2, Command::ON}});
}

void merge_one_microsecond_gap() {
    const auto merged = merge_intervals({{2, 201, 300, {"b"}}, {2, 100, 200, {"a"}}});
    expect_intervals(merged, {{2, 100, 200, {"a"}}, {2, 201, 300, {"b"}}});
    const std::vector<Event> events{{100, 2, Command::ON}, {200, 2, Command::OFF},
                                    {201, 2, Command::ON}, {300, 2, Command::OFF}};
    expect_events(make_schedule(merged), events);
    VirtualExecutor executor{make_schedule(merged)};
    for (const auto& [time, on] :
         std::vector<std::pair<std::int64_t, bool>>{{99, false}, {100, true}, {199, true},
                                                   {200, false}, {201, true}, {299, true},
                                                   {300, false}}) {
        require(executor.advance_to(time) == AdvanceResult::OK, "gap boundary advance");
        expect_on_channels(executor, on ? std::vector<std::size_t>{2} : std::vector<std::size_t>{});
    }
    expect_events(executor.event_log(), events);
}

void merge_identical_and_nested() {
    expect_intervals(merge_intervals({{2, 100, 200, {"z", "a"}}, {2, 100, 200, {"a", "b"}}}),
                     {{2, 100, 200, {"a", "b", "z"}}});
    expect_intervals(merge_intervals({{2, 150, 200, {"inner"}}, {2, 100, 300, {"outer"}},
                                     {2, 100, 250, {"same-start"}}}),
                     {{2, 100, 300, {"inner", "outer", "same-start"}}});
}

void merge_transitive_unsorted() {
    std::vector<MergedInterval> input{{2, 240, 300, {"z"}}, {2, 100, 200, {"c"}},
                                      {2, 150, 250, {"b", "a"}}, {2, 350, 400, {"gap"}}};
    const auto original = input;
    const auto merged = merge_intervals(input);
    expect_intervals(merged, {{2, 100, 300, {"a", "b", "c", "z"}}, {2, 350, 400, {"gap"}}});
    expect_intervals(input, original);
    require(merge_intervals({}).empty() && make_schedule({}).empty(), "empty merge and schedule");
}

void merge_channel_independence() {
    const auto merged = merge_intervals({{7, 100, 200, {"z"}}, {0, 100, 200, {"a"}},
                                         {2, 150, 250, {"c"}}, {0, 200, 300, {"b"}}});
    expect_intervals(merged, {{0, 100, 300, {"a", "b"}}, {2, 150, 250, {"c"}},
                              {7, 100, 200, {"z"}}});
    expect_events(make_schedule(merged), {{100, 0, Command::ON}, {100, 7, Command::ON},
                  {150, 2, Command::ON}, {200, 7, Command::OFF},
                  {250, 2, Command::OFF}, {300, 0, Command::OFF}});
}

void merge_timestamp_limits() {
    const auto merged = merge_intervals({
        {7, max_time - 1, max_time, {"z"}}, {0, 0, max_time - 1, {"a"}},
        {0, max_time - 1, max_time, {"b"}}, {7, 0, max_time - 2, {"early"}}});
    expect_intervals(merged, {{0, 0, max_time, {"a", "b"}},
        {7, 0, max_time - 2, {"early"}}, {7, max_time - 1, max_time, {"z"}}});
    const std::vector<Event> events{
        {0, 0, Command::ON}, {0, 7, Command::ON}, {max_time - 2, 7, Command::OFF},
        {max_time - 1, 7, Command::ON}, {max_time, 0, Command::OFF}, {max_time, 7, Command::OFF}};
    expect_events(make_schedule(merged), events);
    VirtualExecutor executor{make_schedule(merged)};
    require(executor.advance_to(0) == AdvanceResult::OK, "origin events");
    expect_on_channels(executor, {0, 7});
    require(executor.advance_to(max_time - 2) == AdvanceResult::OK, "gap at time limit");
    expect_on_channels(executor, {0});
    require(executor.advance_to(max_time - 1) == AdvanceResult::OK, "gap ends at time limit");
    expect_on_channels(executor, {0, 7});
    require(executor.advance_to(max_time) == AdvanceResult::OK, "complete at time limit");
    expect_on_channels(executor, {});
    expect_events(executor.event_log(), events);
}

void schedule_cross_channel_ties() {
    // Includes ON/ON, OFF/OFF, OFF/ON and ON/OFF ties: channel always breaks ties.
    const std::vector<MergedInterval> intervals{
        {3, 100, 200, {"d"}}, {2, 200, 300, {"c"}},
        {1, 100, 200, {"b"}}, {0, 0, 100, {"a"}}};
    const std::vector<Event> events{
        {0, 0, Command::ON}, {100, 0, Command::OFF},
        {100, 1, Command::ON}, {100, 3, Command::ON},
        {200, 1, Command::OFF}, {200, 2, Command::ON}, {200, 3, Command::OFF},
        {300, 2, Command::OFF}};
    const auto schedule = make_schedule(intervals);
    expect_events(schedule, events);
    VirtualExecutor executor{schedule};
    require(executor.advance_to(100) == AdvanceResult::OK, "tied events at 100");
    expect_on_channels(executor, {1, 3});
    require(executor.advance_to(200) == AdvanceResult::OK, "tied events at 200");
    expect_on_channels(executor, {2});
    require(executor.advance_to(300) == AdvanceResult::OK, "final OFF");
    expect_on_channels(executor, {});
    expect_events(executor.event_log(), events);
}

void batch_advance_repeat_rewind() {
    const std::vector<Event> events{
        {0, 0, Command::ON}, {100, 0, Command::OFF}, {100, 2, Command::ON},
        {150, 7, Command::ON}, {200, 2, Command::OFF}, {201, 2, Command::ON},
        {250, 7, Command::OFF}, {300, 2, Command::OFF}};
    const auto schedule = make_schedule({{0, 0, 100, {"a"}}, {2, 100, 200, {"b"}},
                                        {2, 201, 300, {"c"}}, {7, 150, 250, {"d"}}});
    expect_events(schedule, events);
    VirtualExecutor incremental{schedule};
    VirtualExecutor large{schedule};
    require(incremental.advance_to(-1) == AdvanceResult::CLOCK_REWIND, "initial rewind");
    require(incremental.now_us() == 0 && incremental.event_log().empty(), "initial rewind unchanged");
    expect_on_channels(incremental, {});
    for (const auto time : {0, 99, 100, 150, 199, 200, 201, 250, 300, 999}) {
        require(incremental.advance_to(time) == AdvanceResult::OK, "incremental advance");
        const auto states = incremental.channel_states();
        const auto log = incremental.event_log();
        require(incremental.advance_to(time) == AdvanceResult::OK, "batch repeat time");
        require(incremental.advance_to(time - 1) == AdvanceResult::CLOCK_REWIND, "batch rewind");
        require(incremental.now_us() == time && incremental.channel_states() == states,
                "repeat/rewind preserves time and states");
        expect_events(incremental.event_log(), log);
        std::vector<Event> prefix;
        for (const auto& event : events) {
            if (event.event_time_us <= time) {
                prefix.push_back(event);
            }
        }
        expect_events(log, prefix);
    }
    require(large.advance_to(999) == AdvanceResult::OK, "one large advance");
    expect_events(large.event_log(), events);
    expect_events(incremental.event_log(), events);
    require(large.now_us() == incremental.now_us() &&
            large.channel_states() == incremental.channel_states(), "large/incremental equivalent");
    expect_on_channels(large, {});
    expect_on_channels(incremental, {});
}

void batch_input_permutations() {
    const std::vector<Target> targets{{"a", 0, 0.625, 100.0}, {"b", 0, 0.625, 200.0},
        {"c", 0, 0.625, 250.0}, {"d", 0, 0.875, 200.0}, {"reject", 0, 2.0, 100.0}};
    const std::vector<Pulse> expected_pulses{
        {"a", 2, 100, 100, 200}, {"b", 2, 200, 200, 300},
        {"c", 2, 250, 250, 350}, {"d", 3, 200, 200, 300}};
    const std::vector<MergedInterval> intervals{
        {2, 100, 350, {"a", "b", "c"}}, {3, 200, 300, {"d"}}};
    const std::vector<Event> events{{100, 2, Command::ON}, {200, 3, Command::ON},
                                    {300, 3, Command::OFF}, {350, 2, Command::OFF}};
    std::vector<std::size_t> order{0, 1, 2, 3, 4};
    std::size_t permutations = 0;
    do {
        std::vector<Target> input;
        for (const auto i : order) {
            input.push_back(targets[i]);
        }
        const auto original = input;
        const auto batch = expect_batch(plan_batch(input, {0.25, 1'000'000.0, 0, 100}, 0));
        expect_targets(input, original);
        require(batch.target_results.size() == 5, "permutation result count");
        for (std::size_t i = 0; i < order.size(); ++i) {
            if (order[i] == 4) {
                expect_result_rejection(batch.target_results[i], "reject", RejectionReason::OUT_OF_SWATH);
            } else {
                expect_pulse_record(expect_pulse(batch.target_results[i]), expected_pulses[order[i]]);
            }
        }
        expect_intervals(batch.merged_intervals, intervals);
        expect_events(batch.schedule, events);
        VirtualExecutor executor{batch.schedule};
        require(executor.event_log().empty(), "permutation constructor does not execute");
        require(executor.advance_to(400) == AdvanceResult::OK, "permutation execute");
        expect_events(executor.event_log(), events);
        expect_on_channels(executor, {});
        ++permutations;
    } while (std::next_permutation(order.begin(), order.end()));
    require(permutations == 120, "all input permutations checked");
}

void batch_demo_reference() {
    const auto batch = expect_batch(plan_batch({
        {"b", 1'000'000, 0.625, 1.25}, {"a", 1'000'000, 0.625, 1.0},
        {"c", 1'000'000, 0.875, 1.0}, {"outside", 1'000'000, 2.0, 1.0},
        {"late", 1'000'000, 0.0, 0.0}}, {0.25, 2.0, 50'000, 125'000}, 1'100'000));
    require(batch.target_results.size() == 5, "demo result count");
    expect_pulse_record(expect_pulse(batch.target_results[0]), {"b", 2, 1'625'000, 1'575'000, 1'700'000});
    expect_pulse_record(expect_pulse(batch.target_results[1]), {"a", 2, 1'500'000, 1'450'000, 1'575'000});
    expect_pulse_record(expect_pulse(batch.target_results[2]), {"c", 3, 1'500'000, 1'450'000, 1'575'000});
    expect_result_rejection(batch.target_results[3], "outside", RejectionReason::OUT_OF_SWATH);
    expect_result_rejection(batch.target_results[4], "late", RejectionReason::TOO_LATE);
    expect_intervals(batch.merged_intervals, {{2, 1'450'000, 1'700'000, {"a", "b"}},
                                             {3, 1'450'000, 1'575'000, {"c"}}});
    const std::vector<Event> events{{1'450'000, 2, Command::ON}, {1'450'000, 3, Command::ON},
                                    {1'575'000, 3, Command::OFF}, {1'700'000, 2, Command::OFF}};
    expect_events(batch.schedule, events);
    VirtualExecutor executor{batch.schedule};
    require(executor.advance_to(1'800'000) == AdvanceResult::OK, "demo complete");
    expect_events(executor.event_log(), events);
    expect_on_channels(executor, {});
}

} // namespace

int main() {
    const std::pair<const char*, void (*)()> tests[] = {
        {"reference_plan", reference_plan},
        {"swath_examples", swath_examples},
        {"adjacent_boundary", adjacent_boundary},
        {"ceil_travel", ceil_travel},
        {"zero_forward", zero_forward},
        {"deadline_equality", deadline_equality},
        {"old_capture", old_capture},
        {"invalid_config", invalid_config},
        {"nonfinite_target", nonfinite_target},
        {"behind_bar", behind_bar},
        {"invalid_timestamps", invalid_timestamps},
        {"time_range", time_range},
        {"representable_time_edges", representable_time_edges},
        {"half_open_states", half_open_states},
        {"skip_events", skip_events},
        {"repeat_time", repeat_time},
        {"rewind_time", rewind_time},
        {"validation_precedence", validation_precedence},
        {"finite_geometry_extremes", finite_geometry_extremes},
        {"planner_is_pure", planner_is_pure},
        {"batch_empty", batch_empty},
        {"batch_shared_errors", batch_shared_errors},
        {"batch_singleton", batch_singleton},
        {"batch_mixed_rejections", batch_mixed_rejections},
        {"batch_duplicate_ids", batch_duplicate_ids},
        {"batch_identical_geometry", batch_identical_geometry},
        {"batch_unique_validation_order", batch_unique_validation_order},
        {"batch_time_limits", batch_time_limits},
        {"batch_planning_purity", batch_planning_purity},
        {"merge_overlap", merge_overlap},
        {"merge_touch", merge_touch},
        {"merge_one_microsecond_gap", merge_one_microsecond_gap},
        {"merge_identical_and_nested", merge_identical_and_nested},
        {"merge_transitive_unsorted", merge_transitive_unsorted},
        {"merge_channel_independence", merge_channel_independence},
        {"merge_timestamp_limits", merge_timestamp_limits},
        {"schedule_cross_channel_ties", schedule_cross_channel_ties},
        {"batch_advance_repeat_rewind", batch_advance_repeat_rewind},
        {"batch_input_permutations", batch_input_permutations},
        {"batch_demo_reference", batch_demo_reference},
    };
    std::size_t failed = 0;
    for (const auto& [name, test] : tests) {
        try {
            test();
            std::cout << "PASS " << name << '\n';
        } catch (const std::exception& error) {
            ++failed;
            std::cerr << "FAIL " << name << ": " << error.what() << '\n';
        }
    }
    std::cout << "Scenarios: " << std::size(tests) << ", passed: "
              << std::size(tests) - failed << ", failed: " << failed << '\n';
    return failed == 0 ? 0 : 1;
}

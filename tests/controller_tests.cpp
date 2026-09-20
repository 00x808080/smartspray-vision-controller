#include "controller.hpp"

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

#include "controller.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <tuple>
#include <utility>

namespace smartspray {

namespace {
bool valid_config(const Config& config) {
    const double width_m = nozzle_count * config.nozzle_pitch_m;
    return std::isfinite(config.nozzle_pitch_m) && config.nozzle_pitch_m > 0.0 &&
           std::isfinite(config.speed_mps) && config.speed_mps > 0.0 &&
           std::isfinite(width_m) &&
           config.actuator_delay_us >= 0 && config.pulse_duration_us > 0;
}
} // namespace

PlanResult plan_pulse(const Target& target, const Config& config, std::int64_t now_us) {
    const auto reject = [&](RejectionReason reason) -> PlanResult {
        return Rejection{target.target_id, reason};
    };

    if (!valid_config(config)) {
        return reject(RejectionReason::INVALID_CONFIG);
    }
    const double width_m = nozzle_count * config.nozzle_pitch_m;
    if (now_us < 0 || target.capture_time_us < 0 || target.capture_time_us > now_us) {
        return reject(RejectionReason::INVALID_TIMESTAMP);
    }
    if (!std::isfinite(target.x_m) || !std::isfinite(target.forward_m)) {
        return reject(RejectionReason::INVALID_TARGET);
    }
    if (target.x_m < 0.0 || target.x_m >= width_m) {
        return reject(RejectionReason::OUT_OF_SWATH);
    }
    if (target.forward_m < 0.0) {
        return reject(RejectionReason::BEHIND_BAR);
    }

    const double index = std::floor(target.x_m / config.nozzle_pitch_m);
    if (!std::isfinite(index) || index < 0.0 || index >= nozzle_count) {
        return reject(RejectionReason::INVALID_TARGET);
    }

    // Widen before multiplication to avoid avoidable double intermediate overflow.
    const long double travel_us = std::ceil(
        1'000'000.0L * static_cast<long double>(target.forward_m) /
        static_cast<long double>(config.speed_mps));
    // INT64_MAX can round up when represented in a floating-point type.
    // The exact exclusive bound 2^63 keeps the subsequent cast safe.
    const long double time_limit = std::ldexp(1.0L, 63);
    if (!std::isfinite(travel_us) || travel_us < 0.0L || travel_us >= time_limit) {
        return reject(RejectionReason::TIME_OUT_OF_RANGE);
    }
    const auto travel_time_us = static_cast<std::int64_t>(travel_us);
    constexpr auto max_time = std::numeric_limits<std::int64_t>::max();
    if (target.capture_time_us > max_time - travel_time_us) {
        return reject(RejectionReason::TIME_OUT_OF_RANGE);
    }
    const auto arrival_time_us = target.capture_time_us + travel_time_us;
    // Both operands are in [0, INT64_MAX], so subtraction cannot underflow.
    const auto on_time_us = arrival_time_us - config.actuator_delay_us;
    if (on_time_us > max_time - config.pulse_duration_us) {
        return reject(RejectionReason::TIME_OUT_OF_RANGE);
    }
    const auto off_time_us = on_time_us + config.pulse_duration_us;
    if (on_time_us < now_us) {
        return reject(RejectionReason::TOO_LATE);
    }

    return Pulse{target.target_id, static_cast<std::size_t>(index),
                 arrival_time_us, on_time_us, off_time_us};
}

const char* reason_name(RejectionReason reason) {
    switch (reason) {
    case RejectionReason::INVALID_CONFIG: return "INVALID_CONFIG";
    case RejectionReason::INVALID_TIMESTAMP: return "INVALID_TIMESTAMP";
    case RejectionReason::INVALID_TARGET: return "INVALID_TARGET";
    case RejectionReason::OUT_OF_SWATH: return "OUT_OF_SWATH";
    case RejectionReason::BEHIND_BAR: return "BEHIND_BAR";
    case RejectionReason::TIME_OUT_OF_RANGE: return "TIME_OUT_OF_RANGE";
    case RejectionReason::TOO_LATE: return "TOO_LATE";
    case RejectionReason::DUPLICATE_TARGET_ID: return "DUPLICATE_TARGET_ID";
    }
    return "UNKNOWN";
}

std::vector<MergedInterval> merge_intervals(std::vector<MergedInterval> intervals) {
    std::sort(intervals.begin(), intervals.end(), [](const auto& a, const auto& b) {
        return std::tie(a.nozzle_index, a.on_time_us, a.off_time_us) <
               std::tie(b.nozzle_index, b.on_time_us, b.off_time_us);
    });
    std::vector<MergedInterval> merged;
    for (auto& interval : intervals) {
        if (!merged.empty() && merged.back().nozzle_index == interval.nozzle_index &&
            interval.on_time_us <= merged.back().off_time_us) {
            auto& current = merged.back();
            current.off_time_us = std::max(current.off_time_us, interval.off_time_us);
            current.source_target_ids.insert(current.source_target_ids.end(),
                interval.source_target_ids.begin(), interval.source_target_ids.end());
        } else {
            merged.push_back(std::move(interval));
        }
    }
    for (auto& interval : merged) {
        auto& ids = interval.source_target_ids;
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    }
    return merged;
}

std::vector<Event> make_schedule(const std::vector<MergedInterval>& intervals) {
    std::vector<Event> schedule;
    for (const auto& interval : intervals) {
        schedule.push_back({interval.on_time_us, interval.nozzle_index, Command::ON});
        schedule.push_back({interval.off_time_us, interval.nozzle_index, Command::OFF});
    }
    std::sort(schedule.begin(), schedule.end(), [](const auto& a, const auto& b) {
        return std::tie(a.event_time_us, a.nozzle_index) <
               std::tie(b.event_time_us, b.nozzle_index);
    });
    return schedule;
}

BatchResult plan_batch(const std::vector<Target>& targets, const Config& config,
                       std::int64_t now_us) {
    if (!valid_config(config)) {
        return BatchFailure{RejectionReason::INVALID_CONFIG};
    }
    if (now_us < 0) {
        return BatchFailure{RejectionReason::INVALID_TIMESTAMP};
    }

    std::vector<std::string> ids;
    for (const auto& target : targets) {
        ids.push_back(target.target_id);
    }
    std::sort(ids.begin(), ids.end());

    BatchPlan batch;
    std::vector<MergedInterval> intervals;
    for (const auto& target : targets) {
        const auto range = std::equal_range(ids.begin(), ids.end(), target.target_id);
        // The ID is present: every occurrence of a repeated ID is rejected.
        if (std::next(range.first) != range.second) {
            batch.target_results.push_back(
                Rejection{target.target_id, RejectionReason::DUPLICATE_TARGET_ID});
            continue;
        }
        batch.target_results.push_back(plan_pulse(target, config, now_us));
        if (const auto* pulse = std::get_if<Pulse>(&batch.target_results.back())) {
            intervals.push_back({pulse->nozzle_index, pulse->on_time_us,
                                 pulse->off_time_us, {pulse->target_id}});
        }
    }
    batch.merged_intervals = merge_intervals(std::move(intervals));
    batch.schedule = make_schedule(batch.merged_intervals);
    return batch;
}

VirtualExecutor::VirtualExecutor(const Pulse& pulse)
    : VirtualExecutor(std::vector<Event>{{pulse.on_time_us, pulse.nozzle_index, Command::ON},
                                        {pulse.off_time_us, pulse.nozzle_index, Command::OFF}}) {}

VirtualExecutor::VirtualExecutor(std::vector<Event> schedule)
    : schedule_(std::move(schedule)) {
    states_.fill(Command::OFF);
    events_.reserve(schedule_.size());
}

AdvanceResult VirtualExecutor::advance_to(std::int64_t time_us) {
    if (time_us < now_us_) {
        return AdvanceResult::CLOCK_REWIND;
    }
    while (next_event_ < schedule_.size()) {
        const auto& event = schedule_[next_event_];
        if (event.event_time_us > time_us) {
            break;
        }
        events_.push_back(event);
        states_[event.nozzle_index] = event.command;
        ++next_event_;
    }
    now_us_ = time_us;
    return AdvanceResult::OK;
}

} // namespace smartspray

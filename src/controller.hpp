#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace smartspray {

inline constexpr std::size_t nozzle_count = 8;

struct Target {
    std::string target_id;
    std::int64_t capture_time_us;
    double x_m;
    double forward_m;
};

struct Config {
    double nozzle_pitch_m;
    double speed_mps;
    std::int64_t actuator_delay_us;
    std::int64_t pulse_duration_us;
};

struct Pulse {
    std::string target_id;
    std::size_t nozzle_index;
    std::int64_t arrival_time_us;
    std::int64_t on_time_us;
    std::int64_t off_time_us;
};

enum class RejectionReason {
    INVALID_CONFIG,
    INVALID_TIMESTAMP,
    INVALID_TARGET,
    OUT_OF_SWATH,
    BEHIND_BAR,
    TIME_OUT_OF_RANGE,
    TOO_LATE,
    DUPLICATE_TARGET_ID
};

struct Rejection {
    std::string target_id;
    RejectionReason reason;
};

using PlanResult = std::variant<Pulse, Rejection>;

PlanResult plan_pulse(const Target& target, const Config& config, std::int64_t now_us);
const char* reason_name(RejectionReason reason);

enum class Command { OFF, ON };

struct Event {
    std::int64_t event_time_us;
    std::size_t nozzle_index;
    Command command;
};

struct MergedInterval {
    std::size_t nozzle_index;
    std::int64_t on_time_us;
    std::int64_t off_time_us;
    std::vector<std::string> source_target_ids;
};

// Preconditions: channels in [0, 8), 0 <= ON < OFF, nonempty source IDs list.
// Takes a copy; returns channel/ON-ordered intervals and sorted unique source IDs.
std::vector<MergedInterval> merge_intervals(std::vector<MergedInterval> intervals);

// Precondition: valid intervals, with no overlap or touching on the same channel.
// Returns the complete schedule ordered by (event_time_us, nozzle_index).
std::vector<Event> make_schedule(const std::vector<MergedInterval>& intervals);

struct BatchPlan {
    std::vector<PlanResult> target_results; // Original Pulses/rejections in input order.
    std::vector<MergedInterval> merged_intervals;
    std::vector<Event> schedule;
};

struct BatchFailure {
    RejectionReason reason; // Only INVALID_CONFIG or INVALID_TIMESTAMP.
};

using BatchResult = std::variant<BatchPlan, BatchFailure>;

// Config, then shared now, then all duplicate IDs, then plan_pulse for unique IDs.
BatchResult plan_batch(const std::vector<Target>& targets, const Config& config,
                       std::int64_t now_us);

enum class AdvanceResult { OK, CLOCK_REWIND };

class VirtualExecutor {
public:
    // Precondition: pulse is a successful plan_pulse result, unchanged.
    // The virtual clock starts at zero; execution begins only on advance_to.
    explicit VirtualExecutor(const Pulse& pulse);

    // Precondition: a complete schedule from an unchanged successful BatchPlan,
    // or equivalently sorted (time, channel), nonnegative events for channels [0, 8),
    // strictly alternating ON/OFF per channel, starting ON and ending OFF,
    // with positive intervals and positive gaps. Empty is valid.
    // Owns an immutable copy; construction emits no events.
    explicit VirtualExecutor(std::vector<Event> schedule);

    AdvanceResult advance_to(std::int64_t time_us);
    std::int64_t now_us() const { return now_us_; }
    const std::array<Command, nozzle_count>& channel_states() const { return states_; }
    const std::vector<Event>& event_log() const { return events_; }

private:
    const std::vector<Event> schedule_;
    std::int64_t now_us_ = 0;
    std::array<Command, nozzle_count> states_{};
    std::size_t next_event_ = 0;
    std::vector<Event> events_;
};

} // namespace smartspray

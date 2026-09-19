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
    TOO_LATE
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

enum class AdvanceResult { OK, CLOCK_REWIND };

class VirtualExecutor {
public:
    // Precondition: pulse is a successful plan_pulse result, unchanged.
    // The virtual clock starts at zero; execution begins only on advance_to.
    explicit VirtualExecutor(const Pulse& pulse);

    AdvanceResult advance_to(std::int64_t time_us);
    std::int64_t now_us() const { return now_us_; }
    const std::array<Command, nozzle_count>& channel_states() const { return states_; }
    const std::vector<Event>& event_log() const { return events_; }

private:
    Pulse pulse_;
    std::int64_t now_us_ = 0;
    std::array<Command, nozzle_count> states_{};
    std::size_t next_event_ = 0;
    std::vector<Event> events_;
};

} // namespace smartspray

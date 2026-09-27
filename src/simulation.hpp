#pragma once
#include "controller.hpp"
#include <nlohmann/json.hpp>

namespace smartspray::simulation {
// Complete M3 profile. Wall-clock durations never enter these virtual times.
struct Profile {
    Config controller{0.25, 2.0, 50'000, 100'000};
    double lookahead_m = 2.0;
    std::int64_t capture_time_us = 1'000'000;
    std::int64_t simulated_processing_delay_us = 100'000;
};
std::int64_t validate_profile(const Profile& profile);
Profile parse_profile(const nlohmann::json& value);
nlohmann::json profile_json(const Profile& profile);
nlohmann::json geometry_json(const Profile& profile = {});
nlohmann::json execute_targets(const std::vector<Target>& targets, const Profile& profile = {});
// M3 original-image positive-extent box center mapping; match_id is supplied by caller.
nlohmann::json map_and_execute(const nlohmann::json& detections, const Profile& profile = {});
} // namespace smartspray::simulation

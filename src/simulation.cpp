#include "simulation.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>

namespace smartspray::simulation {
using nlohmann::json;
namespace {
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
std::int64_t integer(const json& value) {
    require(value.is_number_integer(), "Time and schema fields must be integers");
    if (value.is_number_unsigned()) {
        require(value.get<std::uint64_t>() <=
                static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()),
                "Integer exceeds int64 range");
    }
    return value.get<std::int64_t>();
}
}
std::int64_t validate_profile(const Profile& profile) {
    const auto shared = plan_batch({}, profile.controller, 0);
    require(std::holds_alternative<BatchPlan>(shared), "INVALID_CONFIG: controller settings");
    require(std::isfinite(profile.lookahead_m) && profile.lookahead_m > 0.0,
            "INVALID_CONFIG: lookahead_m must be finite and positive");
    require(profile.capture_time_us >= 0 && profile.simulated_processing_delay_us >= 0,
            "INVALID_CONFIG: virtual timestamps/delays must be nonnegative");
    require(profile.capture_time_us <= std::numeric_limits<std::int64_t>::max() -
                profile.simulated_processing_delay_us, "INVALID_CONFIG: now_us overflow");
    return profile.capture_time_us + profile.simulated_processing_delay_us;
}
Profile parse_profile(const json& value) {
    const std::set<std::string> fields{"schema_version", "nozzle_pitch_m", "speed_mps",
        "actuator_delay_us", "pulse_duration_us", "lookahead_m", "capture_time_us",
        "simulated_processing_delay_us"};
    require(value.is_object() && value.size() == fields.size(), "Invalid configuration fields");
    for (const auto& key : fields) require(value.contains(key), "Missing configuration field: " + key);
    require(integer(value.at("schema_version")) == 1, "Unsupported configuration schema");
    for (const char* key : {"nozzle_pitch_m", "speed_mps", "lookahead_m"})
        require(value.at(key).is_number(), std::string("Expected numeric field: ") + key);
    Profile result{{value.at("nozzle_pitch_m").get<double>(), value.at("speed_mps").get<double>(),
                    integer(value.at("actuator_delay_us")), integer(value.at("pulse_duration_us"))},
                   value.at("lookahead_m").get<double>(), integer(value.at("capture_time_us")),
                   integer(value.at("simulated_processing_delay_us"))};
    (void)validate_profile(result);
    return result;
}
json profile_json(const Profile& p) {
    (void)validate_profile(p);
    return {{"schema_version", 1}, {"nozzle_pitch_m", p.controller.nozzle_pitch_m},
            {"speed_mps", p.controller.speed_mps}, {"actuator_delay_us", p.controller.actuator_delay_us},
            {"pulse_duration_us", p.controller.pulse_duration_us}, {"lookahead_m", p.lookahead_m},
            {"capture_time_us", p.capture_time_us},
            {"simulated_processing_delay_us", p.simulated_processing_delay_us}};
}
const char* command_name(Command command) {
    return command == Command::ON ? "ON" : "OFF";
}

json geometry_json(const Profile& profile) {
    const auto config = profile.controller;
    const auto width_m = nozzle_count * config.nozzle_pitch_m;
    const auto lookahead_m = profile.lookahead_m;
    const auto capture_time_us = profile.capture_time_us;
    const auto now_us = validate_profile(profile);
    return {{"width_m", width_m}, {"lookahead_m", lookahead_m},
            {"nozzle_pitch_m", config.nozzle_pitch_m},
            {"speed_mps", config.speed_mps},
            {"capture_time_us", capture_time_us}, {"now_us", now_us},
            {"actuator_delay_us", config.actuator_delay_us},
            {"pulse_duration_us", config.pulse_duration_us},
            {"anchor", "weed_bbox_center_original_pixels"},
            {"coordinate_rounding", "none"}};
}

json event_json(const Event& event) {
    return {{"event_time_us", event.event_time_us},
            {"nozzle_index", event.nozzle_index},
            {"command", command_name(event.command)}};
}

json execute_targets(const std::vector<Target>& targets, const Profile& profile) {
    const auto now_us = validate_profile(profile);
    const auto config = profile.controller;
    json output{{"targets", json::array()}, {"target_results", json::array()},
                {"merged_intervals", json::array()}, {"schedule", json::array()},
                {"executed_events", json::array()}, {"final_states", json::array()}};
    for (const auto& target : targets) {
        output["targets"].push_back({{"target_id", target.target_id},
            {"capture_time_us", target.capture_time_us},
            {"x_m", target.x_m}, {"forward_m", target.forward_m}});
    }
    const auto result = plan_batch(targets, config, now_us);
    if (const auto* failure = std::get_if<BatchFailure>(&result)) {
        throw std::runtime_error(std::string("Unexpected BatchFailure: ") +
                                 reason_name(failure->reason));
    }
    const auto& plan = std::get<BatchPlan>(result);
    for (const auto& target_result : plan.target_results) {
        if (const auto* pulse = std::get_if<Pulse>(&target_result)) {
            output["target_results"].push_back({{"target_id", pulse->target_id},
                {"status", "pulse"}, {"nozzle_index", pulse->nozzle_index},
                {"arrival_time_us", pulse->arrival_time_us},
                {"on_time_us", pulse->on_time_us}, {"off_time_us", pulse->off_time_us}});
        } else {
            const auto& rejection = std::get<Rejection>(target_result);
            output["target_results"].push_back({{"target_id", rejection.target_id},
                {"status", "rejected"}, {"reason", reason_name(rejection.reason)}});
        }
    }
    for (const auto& interval : plan.merged_intervals) {
        output["merged_intervals"].push_back({{"nozzle_index", interval.nozzle_index},
            {"on_time_us", interval.on_time_us}, {"off_time_us", interval.off_time_us},
            {"source_target_ids", interval.source_target_ids}});
    }
    for (const auto& event : plan.schedule) output["schedule"].push_back(event_json(event));
    VirtualExecutor executor(plan.schedule);
    const auto final_time = plan.schedule.empty() ? now_us : std::max(now_us, plan.schedule.back().event_time_us);
    require(executor.advance_to(final_time) == AdvanceResult::OK, "Unexpected executor failure");
    for (const auto& event : executor.event_log()) {
        output["executed_events"].push_back(event_json(event));
    }
    for (const auto state : executor.channel_states()) {
        output["final_states"].push_back(command_name(state));
        require(state == Command::OFF, "Controller did not finish with every channel OFF");
    }
    require(output["schedule"] == output["executed_events"], "Schedule execution mismatch");
    output["final_time_us"] = executor.now_us();
    return output;
}

json map_and_execute(const json& path, const Profile& profile) {
    (void)validate_profile(profile);
    const auto width_m = nozzle_count * profile.controller.nozzle_pitch_m;
    const auto lookahead_m = profile.lookahead_m;
    const auto capture_time_us = profile.capture_time_us;
    require(path.is_object(), "Each named path must be an object");
    const auto image_width = path.at("original_width").get<double>();
    const auto image_height = path.at("original_height").get<double>();
    require(std::isfinite(image_width) && std::isfinite(image_height) &&
            image_width > 0.0 && image_height > 0.0 &&
            image_width <= std::numeric_limits<int>::max() && image_height <= std::numeric_limits<int>::max() &&
            std::floor(image_width) == image_width && std::floor(image_height) == image_height,
            "Original image dimensions must be positive integers");
    const auto& detections = path.at("detections");
    require(detections.is_array(), "detections must be an array");
    std::vector<Target> targets;
    json anchors = json::array();
    std::size_t crops = 0;
    for (const auto& detection : detections) {
        require(detection.at("class_id").is_number_integer(), "class_id must be an integer");
        const auto class_id = detection.at("class_id").get<int>();
        require(class_id == 0 || class_id == 1, "Only crop=0 and weed=1 are accepted");
        if (class_id == 0) { ++crops; continue; }
        const auto id = detection.at("match_id").get<std::string>();
        require(!id.empty(), "Weed match_id must be nonempty; array order is not identity");
        const auto& box = detection.at("xyxy");
        require(box.is_array() && box.size() == 4, "xyxy must contain four original-pixel coordinates");
        const double x1 = box.at(0).get<double>(), y1 = box.at(1).get<double>();
        const double x2 = box.at(2).get<double>(), y2 = box.at(3).get<double>();
        require(std::isfinite(x1) && std::isfinite(y1) && std::isfinite(x2) && std::isfinite(y2),
                "Box coordinates must be finite");
        require(0.0 <= x1 && x1 < x2 && x2 <= image_width &&
                0.0 <= y1 && y1 < y2 && y2 <= image_height,
                "Boxes must have positive extents within the original image");
        const double u_px = (x1 + x2) / 2.0;
        const double v_px = (y1 + y2) / 2.0;
        const double x_m = width_m * u_px / image_width;
        const double forward_m = lookahead_m * (1.0 - v_px / image_height);
        targets.push_back({id, capture_time_us, x_m, forward_m});
        anchors.push_back({{"target_id", id}, {"u_px", u_px}, {"v_px", v_px}, {"xyxy", box}});
    }
    auto output = execute_targets(targets, profile);
    output["anchors"] = std::move(anchors);
    output["original_width"] = image_width;
    output["original_height"] = image_height;
    output["crop_detections_ignored"] = crops;
    return output;
}

} // namespace smartspray::simulation

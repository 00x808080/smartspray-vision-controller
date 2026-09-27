#include "demo.hpp"
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <set>
#include <sstream>
#include <stdexcept>

namespace smartspray::demo {
using nlohmann::json;
json compose(const std::vector<vision::Detection>& detections,
             const vision::Geometry& geometry, const simulation::Profile& profile) {
    const auto now = simulation::validate_profile(profile);
    auto predictions = vision::detection_json(detections, geometry);
    std::set<std::size_t> ids;
    for (auto& d : predictions["detections"]) {
        const auto candidate = d.at("candidate_index").get<std::size_t>();
        if (!ids.insert(candidate).second || d.at("score").get<float>() <= vision::confidence_threshold)
            throw std::runtime_error("Invalid post-NMS detection identity/confidence");
        d["match_id"] = "t" + std::to_string(candidate);
    }
    auto action = simulation::map_and_execute(predictions, profile);
    json trace = json::array();
    std::size_t selected = 0, accepted = 0;
    for (auto d : predictions.at("detections")) {
        const std::string target_id = d.at("match_id");
        d.erase("match_id");
        const bool weed = d.at("class_id") == 1;
        json record{{"detection_id", "d" + std::to_string(d.at("candidate_index").get<std::size_t>())},
                    {"prediction", d}, {"selected_for_mapping", weed}};
        if (!weed) {
            record["status"] = "not_selected";
            record["reason"] = "predicted_crop";
        } else {
            record["target_id"] = target_id;
            record["anchor"] = action.at("anchors").at(selected);
            record["target"] = action.at("targets").at(selected);
            record["plan"] = action.at("target_results").at(selected++);
            const bool pulse = record["plan"]["status"] == "pulse";
            record["status"] = pulse ? "accepted" : "rejected";
            if (pulse) ++accepted;
        }
        trace.push_back(std::move(record));
    }
    return {{"schema_version", 1}, {"status", "completed"}, {"simulation_only", true},
        {"configuration", simulation::profile_json(profile)},
        {"simulation", {{"swath_width_m", nozzle_count * profile.controller.nozzle_pitch_m},
            {"now_us", now}, {"channels", nozzle_count},
            {"anchor", "predicted_weed_bbox_center_original_pixels_not_verified_stem"},
            {"mapping", {{"x_m", "swath_width_m * u_px / image_width_px"},
                         {"forward_m", "lookahead_m * (1 - v_px / image_height_px)"}}},
            {"image_domain", "0 <= u_px < width; 0 <= v_px < height; origin top-left"},
            {"coordinate_rounding", "none; no camera offset or target clamping"},
            {"time", "integer microseconds from shared virtual origin; not wall-clock or UTC"},
            {"command_intervals", "half-open [ON, OFF); fixed equal actuator ON/OFF delay"},
            {"final_time_rule", "max(now_us, every scheduled OFF); no added tick"}}},
        {"inference", {{"confidence_threshold", vision::confidence_threshold},
            {"confidence_comparison", ">"}, {"nms_iou_threshold", vision::nms_iou_threshold},
            {"suppression_comparison", ">"}, {"nms_class_aware", true},
            {"maximum_detections", vision::maximum_detections}, {"class_names", {"crop", "weed"}},
            {"ordering", "score descending, graph candidate index ascending; class tie chooses crop"},
            {"preprocessing", "M3 BGR decode; centered 1024 letterbox, scale-up, INTER_LINEAR, pad114; RGB FP32 /255 NCHW"},
            {"geometry", vision::geometry_json(geometry)}}},
        {"trace", trace}, {"action", action},
        {"counts", {{"detections", detections.size()}, {"crops_not_selected", detections.size()-selected},
            {"weeds_selected", selected}, {"targets", selected}, {"accepted", accepted},
            {"rejected", selected-accepted}, {"merged_intervals", action["merged_intervals"].size()},
            {"events", action["executed_events"].size()}}}};
}
std::string events_csv(const json& run) {
    std::ostringstream out;
    out << "event_time_us,nozzle_index,command\n";
    for (const auto& e : run.at("action").at("executed_events"))
        out << e.at("event_time_us").get<std::int64_t>() << ','
            << e.at("nozzle_index").get<std::size_t>() << ','
            << e.at("command").get<std::string>() << '\n';
    return out.str();
}
void require_new_output(const std::filesystem::path& directory) {
    if (directory.empty() || directory.filename().empty() ||
        std::filesystem::symlink_status(directory).type() != std::filesystem::file_type::not_found)
        throw std::runtime_error("Output directory must not exist: " + directory.string());
    const auto parent = directory.has_parent_path() ? directory.parent_path() : std::filesystem::path(".");
    if (!std::filesystem::is_directory(parent))
        throw std::runtime_error("Output parent must be an existing directory: " + parent.string());
}
void write_outputs(const std::filesystem::path& directory, const json& run,
                   const cv::Mat& annotated, const cv::Mat& timeline) {
    require_new_output(directory);
    // Encode/serialize first: invalid data cannot create a success-looking directory.
    std::vector<unsigned char> annotation_bytes, timeline_bytes;
    if (!cv::imencode(".png", annotated, annotation_bytes) ||
        !cv::imencode(".png", timeline, timeline_bytes))
        throw std::runtime_error("Cannot encode presentation PNGs");
    const auto csv = events_csv(run);
    const auto serialized = run.dump(2) + "\n";
    if (!std::filesystem::create_directory(directory))
        throw std::runtime_error("Output directory already exists");
    try {
        vision::write_binary((directory/"annotated.png").string(), annotation_bytes.data(), annotation_bytes.size());
        vision::write_binary((directory/"timeline.png").string(), timeline_bytes.data(), timeline_bytes.size());
        vision::write_binary((directory/"events.csv").string(), csv.data(), csv.size());
        // A crash before this rename leaves no completed run.json.
        vision::write_binary((directory/"run.json.incomplete").string(), serialized.data(), serialized.size());
        std::filesystem::rename(directory/"run.json.incomplete", directory/"run.json");
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove_all(directory, ignored);
        throw;
    }
}
} // namespace smartspray::demo

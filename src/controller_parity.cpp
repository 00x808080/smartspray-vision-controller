#include "controller.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using nlohmann::json;
using namespace smartspray;
constexpr double width_m = 2.0;
constexpr double lookahead_m = 2.0;
constexpr std::int64_t capture_time_us = 1'000'000;
constexpr std::int64_t now_us = 1'100'000;
const Config config{0.25, 2.0, 50'000, 100'000};

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

const char* command_name(Command command) {
    return command == Command::ON ? "ON" : "OFF";
}

json geometry_json() {
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

json execute_targets(const std::vector<Target>& targets) {
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
        // Frozen valid shared settings make this a harness failure, not a detector result.
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
    const auto final_time = plan.schedule.empty() ? now_us : plan.schedule.back().event_time_us;
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

json map_and_execute(const json& path) {
    require(path.is_object(), "Each named path must be an object");
    const auto image_width = path.at("original_width").get<double>();
    const auto image_height = path.at("original_height").get<double>();
    require(std::isfinite(image_width) && std::isfinite(image_height) &&
            image_width > 0.0 && image_height > 0.0 &&
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
    auto output = execute_targets(targets);
    output["anchors"] = std::move(anchors);
    output["original_width"] = image_width;
    output["original_height"] = image_height;
    output["crop_detections_ignored"] = crops;
    return output;
}

json run_cases(const json& input) {
    const auto& cases = input.at("cases");
    require(cases.is_array(), "cases must be an array");
    json output{{"schema_version", 1}, {"geometry", geometry_json()}, {"cases", json::array()}};
    std::set<std::string> case_ids;
    for (const auto& item : cases) {
        const auto id = item.at("id").get<std::string>();
        require(!id.empty() && case_ids.insert(id).second, "Case IDs must be nonempty and unique");
        const auto& paths = item.at("paths");
        require(paths.is_object() && !paths.empty(), "paths must be a nonempty object");
        json result{{"id", id}, {"paths", json::object()}};
        for (const auto& [name, path] : paths.items()) {
            require(!name.empty(), "Path names must be nonempty");
            result["paths"][name] = map_and_execute(path);
        }
        output["cases"].push_back(std::move(result));
    }
    return output;
}

json self_test() {
    const double infinity = std::numeric_limits<double>::infinity();
    json tests = json::array();
    const auto run = [](const std::string& id, double x, double forward) {
        return execute_targets({Target{id, capture_time_us, x, forward}});
    };
    const auto assert_pulse = [](const json& result, std::size_t channel, std::int64_t on) {
        const auto& target = result.at("target_results").at(0);
        require(target.at("status") == "pulse" && target.at("nozzle_index") == channel &&
                target.at("on_time_us") == on && target.at("off_time_us") == on + 100'000,
                "Boundary pulse assertion failed");
        require(result.at("executed_events").size() == 2, "Expected exactly two executed events");
    };
    const auto assert_rejected = [](const json& result, const std::string& reason) {
        const auto& target = result.at("target_results").at(0);
        require(target.at("status") == "rejected" && target.at("reason") == reason,
                "Boundary rejection assertion failed");
        require(result.at("executed_events").empty(), "Rejected target emitted events");
    };

    auto below = run("boundary", std::nextafter(0.25, -infinity), 1.0);
    auto exact = run("boundary", 0.25, 1.0);
    auto above = run("boundary", std::nextafter(0.25, infinity), 1.0);
    assert_pulse(below, 0, 1'450'000);
    assert_pulse(exact, 1, 1'450'000);
    assert_pulse(above, 1, 1'450'000);
    tests.push_back({{"name", "channel_boundary_nextafter"}, {"passed", true},
        {"expected", "Below 0.25 m selects channel 0; exact and above select channel 1"},
        {"below", below}, {"exact", exact}, {"above", above}});

    below = run("swath", std::nextafter(2.0, -infinity), 1.0);
    exact = run("swath", 2.0, 1.0);
    above = run("swath", std::nextafter(2.0, infinity), 1.0);
    assert_pulse(below, 7, 1'450'000);
    assert_rejected(exact, "OUT_OF_SWATH");
    assert_rejected(above, "OUT_OF_SWATH");
    tests.push_back({{"name", "swath_boundary_nextafter"}, {"passed", true},
        {"expected", "Below 2.0 m selects channel 7; exact and above are OUT_OF_SWATH"},
        {"below", below}, {"exact", exact}, {"above", above}});

    // 0.299998 as binary64 is slightly below the exact 149999 / 500000
    // boundary. Its next higher binary64 value crosses that boundary.
    below = run("lateness", 0.625, std::nextafter(0.299998, -infinity));
    exact = run("lateness", 0.625, 0.299998);
    above = run("lateness", 0.625, std::nextafter(0.299998, infinity));
    assert_rejected(below, "TOO_LATE");
    assert_rejected(exact, "TOO_LATE");
    assert_pulse(above, 2, now_us);
    tests.push_back({{"name", "lateness_ceil_boundary_nextafter"}, {"passed", true},
        {"expected", "Below and binary64 0.299998 are TOO_LATE; nextabove gives ON exactly now"},
        {"below", below}, {"binary64_anchor", exact}, {"above", above}});

    below = run("ceil", 0.625, std::nextafter(0.3, -infinity));
    exact = run("ceil", 0.625, 0.3);
    above = run("ceil", 0.625, std::nextafter(0.3, infinity));
    assert_pulse(below, 2, now_us);
    assert_pulse(exact, 2, now_us);
    assert_pulse(above, 2, now_us + 1);
    tests.push_back({{"name", "arrival_ceil_nextafter"}, {"passed", true},
        {"expected", "Below and binary64 0.3 give ON now; nextabove shifts ON/OFF by 1 us"},
        {"below", below}, {"binary64_anchor", exact}, {"above", above}});

    const json mapped = map_and_execute({{"original_width", 1024}, {"original_height", 1024},
        {"detections", json::array({
            {{"match_id", "selected"}, {"class_id", 1}, {"xyxy", {310.0, 502.0, 330.0, 522.0}}},
            {{"match_id", "crop"}, {"class_id", 0}, {"xyxy", {0.0, 0.0, 10.0, 10.0}}}})}});
    require(mapped.at("targets").size() == 1 && mapped.at("targets").at(0).at("x_m") == 0.625 &&
            mapped.at("targets").at(0).at("forward_m") == 1.0 &&
            mapped.at("targets").at(0).at("target_id") == "selected" &&
            mapped.at("crop_detections_ignored") == 1, "Weed mapping assertion failed");
    assert_pulse(mapped, 2, 1'450'000);
    tests.push_back({{"name", "weed_mapping_and_crop_filter"}, {"passed", true}, {"result", mapped}});

    const auto empty = execute_targets({});
    require(empty.at("target_results").empty() && empty.at("executed_events").empty() &&
            empty.at("final_states") == json::array({"OFF", "OFF", "OFF", "OFF", "OFF", "OFF", "OFF", "OFF"}),
            "Empty target batch assertion failed");
    tests.push_back({{"name", "empty_batch_finishes_off"}, {"passed", true}, {"result", empty}});
    return {{"schema_version", 1}, {"geometry", geometry_json()}, {"passed", true}, {"tests", tests}};
}
} // namespace

int main(int argc, char** argv) {
    try {
        std::string input_path, output_path;
        bool run_self_test = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--self-test") run_self_test = true;
            else if (arg == "--input" && i + 1 < argc) input_path = argv[++i];
            else if (arg == "--output" && i + 1 < argc) output_path = argv[++i];
            else throw std::runtime_error("Usage: controller_parity (--input FILE | --self-test) [--output FILE]");
        }
        require(run_self_test != !input_path.empty(), "Select exactly one of --input or --self-test");
        if (!input_path.empty() && !output_path.empty()) {
            const auto source = std::filesystem::weakly_canonical(input_path);
            const auto destination = std::filesystem::weakly_canonical(output_path);
            require(source != destination && !(std::filesystem::exists(source) &&
                    std::filesystem::exists(destination) &&
                    std::filesystem::equivalent(source, destination)),
                    "Input and output must be distinct files");
        }
        json result;
        if (run_self_test) result = self_test();
        else {
            std::ifstream input(input_path);
            require(input.is_open(), "Cannot open input JSON: " + input_path);
            json request;
            input >> request;
            result = run_cases(request);
        }
        if (output_path.empty()) std::cout << result.dump(2) << '\n';
        else {
            std::ofstream output(output_path);
            require(output.is_open(), "Cannot open output JSON: " + output_path);
            output << result.dump(2) << '\n';
            require(output.good(), "Could not write output JSON: " + output_path);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "controller_parity: " << error.what() << '\n';
        return 1;
    }
}

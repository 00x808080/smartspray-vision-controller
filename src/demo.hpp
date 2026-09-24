#pragma once
#include "simulation.hpp"
#include "vision.hpp"
#include <filesystem>

namespace smartspray::demo {
nlohmann::json compose(const std::vector<vision::Detection>& detections,
                       const vision::Geometry& geometry, const simulation::Profile& profile);
std::string events_csv(const nlohmann::json& run);
cv::Mat render_annotated(const cv::Mat& original, const nlohmann::json& run);
cv::Mat render_timeline(const nlohmann::json& run);
// Requires a nonexistent directory under an existing parent. run.json is written last.
// Only this invocation's newly created directory is removed on a caught write failure.
void write_outputs(const std::filesystem::path& directory, const nlohmann::json& run,
                   const cv::Mat& annotated, const cv::Mat& timeline);
void require_new_output(const std::filesystem::path& directory);
} // namespace smartspray::demo

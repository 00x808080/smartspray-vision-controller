#pragma once

#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace smartspray::vision {

constexpr int input_size = 1024;
constexpr std::size_t input_elements = 3U * input_size * input_size;
constexpr float confidence_threshold = 0.25F;
constexpr float nms_iou_threshold = 0.7F;
constexpr std::size_t maximum_detections = 300;
constexpr const char* class_names[2] = {"crop", "weed"};

struct Geometry {
    int original_width, original_height;
    double r;
    int resized_width, resized_height;
    int left, top, right, bottom;
};

struct PreparedImage {
    cv::Mat bgr;
    Geometry geometry;
    std::vector<float> input;
};

struct RawOutput {
    // Channel-major [1, 6, candidates]: cx, cy, width, height, crop, weed.
    std::vector<float> values;
    std::size_t candidates;
};

struct Detection {
    int class_id;
    float score;
    std::array<float, 4> xyxy;
    std::size_t candidate_index;
};

int round_ties_to_even(double value);
Geometry letterbox_geometry(int width, int height);
cv::Mat decode_image(const std::string& path);
PreparedImage preprocess(const cv::Mat& bgr);
PreparedImage prepare_image(const std::string& path);
void validate_input(const std::vector<float>& input);
std::vector<Detection> postprocess(const RawOutput& raw, const Geometry& geometry,
                                  float confidence = confidence_threshold,
                                  float iou = nms_iou_threshold,
                                  std::size_t max_det = maximum_detections);
nlohmann::json geometry_json(const Geometry& geometry);
nlohmann::json detection_json(const std::vector<Detection>& detections,
                             const Geometry& geometry);
void write_json(const std::string& path, const nlohmann::json& value);
std::vector<float> read_input_tensor(const std::string& path);
void write_binary(const std::string& path, const void* data, std::size_t bytes);

}  // namespace smartspray::vision

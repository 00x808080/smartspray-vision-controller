#include "vision.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace smartspray::vision {
namespace {
void fail(const std::string& message) { throw std::runtime_error(message); }

void validate_geometry(const Geometry& g) {
    const auto expected = letterbox_geometry(g.original_width, g.original_height);
    if (g.r != expected.r || g.resized_width != expected.resized_width ||
        g.resized_height != expected.resized_height || g.left != expected.left ||
        g.top != expected.top || g.right != expected.right || g.bottom != expected.bottom) {
        fail("invalid letterbox geometry");
    }
}

float intersection_over_union(const Detection& a, const Detection& b) {
    const float width = std::max(0.0F, std::min(a.xyxy[2], b.xyxy[2]) -
                                      std::max(a.xyxy[0], b.xyxy[0]));
    const float height = std::max(0.0F, std::min(a.xyxy[3], b.xyxy[3]) -
                                       std::max(a.xyxy[1], b.xyxy[1]));
    const float intersection = width * height;
    const float area_a = (a.xyxy[2] - a.xyxy[0]) * (a.xyxy[3] - a.xyxy[1]);
    const float area_b = (b.xyxy[2] - b.xyxy[0]) * (b.xyxy[3] - b.xyxy[1]);
    const float union_area = area_a + area_b - intersection;
    if (!std::isfinite(union_area) || !std::isfinite(intersection)) {
        fail("box area is outside FP32 range");
    }
    return union_area > 0.0F ? intersection / union_area : 0.0F;
}
}  // namespace

int round_ties_to_even(double value) {
    if (!std::isfinite(value) || value < 0.0 ||
        value > static_cast<double>(std::numeric_limits<int>::max()) - 1.0) {
        fail("rounding input is outside supported range");
    }
    const double floor_value = std::floor(value);
    int result = static_cast<int>(floor_value);
    const double remainder = value - floor_value;
    if (remainder > 0.5 || (remainder == 0.5 && result % 2 != 0)) {
        ++result;
    }
    return result;
}

Geometry letterbox_geometry(int width, int height) {
    if (width <= 0 || height <= 0) fail("image dimensions must be positive");
    const double r = std::min(static_cast<double>(input_size) / width,
                              static_cast<double>(input_size) / height);
    const int resized_width = round_ties_to_even(width * r);
    const int resized_height = round_ties_to_even(height * r);
    if (resized_width <= 0 || resized_height <= 0) {
        fail("image aspect ratio produces a zero resize dimension");
    }
    const int horizontal = input_size - resized_width;
    const int vertical = input_size - resized_height;
    // Equivalent to round(d/2 - 0.1), round(d/2 + 0.1) for integer d.
    return {width, height, r, resized_width, resized_height,
            horizontal / 2, vertical / 2, horizontal - horizontal / 2,
            vertical - vertical / 2};
}

cv::Mat decode_image(const std::string& path) {
    auto extension = std::filesystem::path(path).extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (extension != ".png" && extension != ".bmp") {
        fail("unsupported image format: only three-channel 8-bit PNG/BMP are accepted");
    }
    std::ifstream source(path, std::ios::binary);
    std::array<unsigned char, 8> signature{};
    source.read(reinterpret_cast<char*>(signature.data()), signature.size());
    const std::array<unsigned char, 8> png_signature{137, 80, 78, 71, 13, 10, 26, 10};
    const bool valid_signature = extension == ".png" ?
        (source.gcount() == 8 && signature == png_signature) :
        (source.gcount() >= 2 && signature[0] == 'B' && signature[1] == 'M');
    if (!valid_signature) fail("image file signature does not match supported PNG/BMP format");
    // Unchanged preserves the source channel count/depth and ignores EXIF rotation.
    cv::Mat bgr = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (bgr.empty()) fail("cannot decode image: " + path);
    if (bgr.type() != CV_8UC3 || bgr.dims != 2) {
        fail("image must decode as three-channel 8-bit BGR");
    }
    return bgr;
}

PreparedImage preprocess(const cv::Mat& bgr) {
    if (bgr.empty() || bgr.dims != 2 || bgr.type() != CV_8UC3) {
        fail("preprocessing requires a nonempty three-channel 8-bit BGR image");
    }
    const auto geometry = letterbox_geometry(bgr.cols, bgr.rows);
    cv::Mat resized;
    if (bgr.cols == geometry.resized_width && bgr.rows == geometry.resized_height) {
        resized = bgr;
    } else {
        cv::resize(bgr, resized, cv::Size(geometry.resized_width, geometry.resized_height),
                   0.0, 0.0, cv::INTER_LINEAR);
    }
    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, geometry.top, geometry.bottom,
                       geometry.left, geometry.right, cv::BORDER_CONSTANT,
                       cv::Scalar(114, 114, 114));
    std::vector<float> input(input_elements);
    const auto plane = static_cast<std::size_t>(input_size) * input_size;
    for (int y = 0; y < input_size; ++y) {
        const auto* row = padded.ptr<cv::Vec3b>(y);
        for (int x = 0; x < input_size; ++x) {
            const auto position = static_cast<std::size_t>(y) * input_size + x;
            input[position] = static_cast<float>(row[x][2]) / 255.0F;
            input[plane + position] = static_cast<float>(row[x][1]) / 255.0F;
            input[2 * plane + position] = static_cast<float>(row[x][0]) / 255.0F;
        }
    }
    return {bgr.clone(), geometry, std::move(input)};
}

PreparedImage prepare_image(const std::string& path) { return preprocess(decode_image(path)); }

void validate_input(const std::vector<float>& input) {
    if (input.size() != input_elements) fail("input tensor must have shape [1,3,1024,1024]");
    for (const float value : input) {
        if (!std::isfinite(value) || value < 0.0F || value > 1.0F) {
            fail("input tensor contains non-finite or out-of-range values");
        }
    }
}

std::vector<Detection> postprocess(const RawOutput& raw, const Geometry& geometry,
                                  float confidence, float iou, std::size_t max_det) {
    validate_geometry(geometry);
    if (raw.candidates == 0 || raw.candidates > std::numeric_limits<std::size_t>::max() / 6 ||
        raw.values.size() != 6 * raw.candidates) fail("raw output must have shape [1,6,N], N > 0");
    if (!std::isfinite(confidence) || confidence < 0.0F || confidence > 1.0F ||
        !std::isfinite(iou) || iou < 0.0F || iou > 1.0F || max_det == 0) {
        fail("invalid postprocessing settings");
    }
    for (const float value : raw.values) {
        if (!std::isfinite(value)) fail("raw output contains non-finite values");
    }
    std::vector<Detection> candidates;
    for (std::size_t index = 0; index < raw.candidates; ++index) {
        const float cx = raw.values[index];
        const float cy = raw.values[raw.candidates + index];
        const float width = raw.values[2 * raw.candidates + index];
        const float height = raw.values[3 * raw.candidates + index];
        const float crop = raw.values[4 * raw.candidates + index];
        const float weed = raw.values[5 * raw.candidates + index];
        if (width < 0.0F || height < 0.0F || crop < 0.0F || crop > 1.0F ||
            weed < 0.0F || weed > 1.0F) fail("raw box dimensions or probabilities are out of range");
        const int class_id = weed > crop ? 1 : 0;
        const float score = class_id == 0 ? crop : weed;
        if (!(score > confidence)) continue;
        Detection candidate{class_id, score,
                            {cx - width / 2.0F, cy - height / 2.0F,
                             cx + width / 2.0F, cy + height / 2.0F}, index};
        for (float coordinate : candidate.xyxy) {
            if (!std::isfinite(coordinate)) fail("decoded box is outside FP32 range");
        }
        // Validate area even when the candidate never enters an IoU comparison.
        (void)intersection_over_union(candidate, candidate);
        candidates.push_back(candidate);
    }
    std::sort(candidates.begin(), candidates.end(), [](const Detection& a, const Detection& b) {
        return a.score != b.score ? a.score > b.score : a.candidate_index < b.candidate_index;
    });
    std::vector<Detection> kept;
    for (const auto& candidate : candidates) {
        bool suppressed = false;
        for (const auto& selected : kept) {
            if (candidate.class_id == selected.class_id &&
                intersection_over_union(candidate, selected) > iou) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            kept.push_back(candidate);
            if (kept.size() == max_det) break;
        }
    }
    const float ratio = static_cast<float>(geometry.r);
    for (auto& detection : kept) {
        for (std::size_t axis = 0; axis < 4; ++axis) {
            const bool x_axis = axis % 2 == 0;
            const float pad = static_cast<float>(x_axis ? geometry.left : geometry.top);
            const float bound = static_cast<float>(x_axis ? geometry.original_width :
                                                           geometry.original_height);
            const float restored = (detection.xyxy[axis] - pad) / ratio;
            if (!std::isfinite(restored)) fail("restored box is outside FP32 range");
            detection.xyxy[axis] = std::clamp(restored, 0.0F, bound);
        }
    }
    return kept;
}

nlohmann::json geometry_json(const Geometry& g) {
    validate_geometry(g);
    return {{"original_width", g.original_width}, {"original_height", g.original_height},
            {"r", g.r}, {"resized_width", g.resized_width}, {"resized_height", g.resized_height},
            {"left", g.left}, {"top", g.top}, {"right", g.right}, {"bottom", g.bottom}};
}

nlohmann::json detection_json(const std::vector<Detection>& detections, const Geometry& geometry) {
    validate_geometry(geometry);
    auto result = nlohmann::json{{"schema_version", 1},
                               {"original_width", geometry.original_width},
                               {"original_height", geometry.original_height},
                               {"classes", {{"0", "crop"}, {"1", "weed"}}},
                               {"detections", nlohmann::json::array()}};
    for (const auto& detection : detections) {
        if (detection.class_id < 0 || detection.class_id > 1 ||
            !std::isfinite(detection.score) || detection.score < 0.0F || detection.score > 1.0F) {
            fail("invalid detection for serialization");
        }
        for (std::size_t axis = 0; axis < 4; ++axis) {
            const float coordinate = detection.xyxy[axis];
            const int bound = axis % 2 == 0 ? geometry.original_width : geometry.original_height;
            if (!std::isfinite(coordinate) || coordinate < 0.0F || coordinate > bound) {
                fail("invalid detection coordinates for serialization");
            }
        }
        if (detection.xyxy[0] > detection.xyxy[2] || detection.xyxy[1] > detection.xyxy[3]) {
            fail("inverted detection coordinates for serialization");
        }
        result["detections"].push_back({{"class_id", detection.class_id},
                                        {"class_name", class_names[detection.class_id]},
                                        {"score", detection.score}, {"xyxy", detection.xyxy},
                                        {"candidate_index", detection.candidate_index}});
    }
    return result;
}

void write_json(const std::string& path, const nlohmann::json& value) {
    // Serialize before opening so serialization errors never truncate an existing output.
    const std::string serialized = value.dump(2) + "\n";
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) fail("cannot open JSON output: " + path);
    stream.write(serialized.data(), static_cast<std::streamsize>(serialized.size()));
    stream.close();
    if (!stream) fail("cannot write JSON output: " + path);
}

std::vector<float> read_input_tensor(const std::string& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream || stream.tellg() != static_cast<std::streamoff>(input_elements * sizeof(float))) {
        fail("diagnostic tensor must contain exactly 3*1024*1024 FP32 values");
    }
    stream.seekg(0);
    std::vector<float> result(input_elements);
    stream.read(reinterpret_cast<char*>(result.data()),
                static_cast<std::streamsize>(result.size() * sizeof(float)));
    if (!stream) fail("cannot read diagnostic input tensor");
    validate_input(result);
    return result;
}

void write_binary(const std::string& path, const void* data, std::size_t bytes) {
    if (bytes > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
        fail("binary output is too large");
    }
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) fail("cannot open binary output: " + path);
    stream.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    stream.close();
    if (!stream) fail("cannot write binary output: " + path);
}
}  // namespace smartspray::vision

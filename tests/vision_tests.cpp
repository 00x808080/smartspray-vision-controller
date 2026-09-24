#include "vision_runtime.hpp"
#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
using namespace smartspray::vision;
constexpr float nan_value = std::numeric_limits<float>::quiet_NaN();
constexpr float infinity = std::numeric_limits<float>::infinity();

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template<class Function> void rejects(Function action) {
    try { action(); } catch (const std::exception&) { return; }
    throw std::runtime_error("expected explicit failure");
}

void near(float actual, float expected, float tolerance = 1e-6F) {
    require(std::abs(actual - expected) <= tolerance, "unexpected floating-point value");
}

struct TemporaryDirectory {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("smartspray-vision-tests-" + std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()));
    TemporaryDirectory() {
        if (!std::filesystem::create_directory(path)) throw std::runtime_error("temporary directory exists");
    }
    ~TemporaryDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }
};

RawOutput output(std::initializer_list<std::array<float, 6>> candidates) {
    RawOutput raw{{}, candidates.size()};
    raw.values.resize(candidates.size() * 6);
    std::size_t index = 0;
    for (const auto& candidate : candidates) {
        for (std::size_t channel = 0; channel < 6; ++channel) {
            raw.values[channel * raw.candidates + index] = candidate[channel];
        }
        ++index;
    }
    return raw;
}

void rounding_ties() {
    require(round_ties_to_even(0.5) == 0 && round_ties_to_even(1.5) == 2 &&
            round_ties_to_even(2.5) == 2 && round_ties_to_even(3.5) == 4, "ties to even");
    require(round_ties_to_even(std::nextafter(2.5, 0.0)) == 2 &&
            round_ties_to_even(std::nextafter(2.5, 3.0)) == 3, "neighbors of tie");
    rejects([] { round_ties_to_even(-0.1); });
    rejects([] { round_ties_to_even(infinity); });
    rejects([] { round_ties_to_even(nan_value); });
    rejects([] { round_ties_to_even(std::numeric_limits<double>::max()); });
}

void geometry_odd_and_ties() {
    auto g = letterbox_geometry(333, 1000);
    require(g.resized_width == 341 && g.resized_height == 1024 &&
            g.left == 341 && g.right == 342 && g.top == 0 && g.bottom == 0, "odd padding");
    require(g.r == 1.024, "scale-up policy");
    g = letterbox_geometry(2048, 5);
    require(g.resized_height == 2 && g.top == 511 && g.bottom == 511, "even resize tie");
    g = letterbox_geometry(2048, 7);
    require(g.resized_height == 4 && g.top == 510, "odd resize tie");
    g = letterbox_geometry(1000, 333);
    require(g.top == 341 && g.bottom == 342, "vertical odd padding");
    rejects([] { letterbox_geometry(0, 10); });
    rejects([] { letterbox_geometry(-1, 10); });
    rejects([] { letterbox_geometry(1, std::numeric_limits<int>::max()); });
}

void rgb_normalization_and_layout() {
    cv::Mat image(input_size, input_size, CV_8UC3, cv::Scalar(7, 29, 255));
    image.at<cv::Vec3b>(0, 1) = {128, 0, 1};
    const auto prepared = preprocess(image);
    const auto plane = static_cast<std::size_t>(input_size) * input_size;
    require(prepared.input.size() == input_elements, "NCHW element count");
    near(prepared.input[0], 1.0F);
    near(prepared.input[plane], 29.0F / 255.0F);
    near(prepared.input[2 * plane], 7.0F / 255.0F);
    near(prepared.input[1], 1.0F / 255.0F);
    near(prepared.input[plane + 1], 0.0F);
    near(prepared.input[2 * plane + 1], 128.0F / 255.0F);
    require(prepared.geometry.r == 1.0 && prepared.geometry.left == 0, "no resize");
    validate_input(prepared.input);
}

void padding_and_noncontiguous_image() {
    cv::Mat allocation(1000, 335, CV_8UC3, cv::Scalar(0, 0, 255));
    const cv::Mat region = allocation(cv::Rect(1, 0, 333, 1000));
    require(!region.isContinuous(), "fixture must be non-contiguous");
    const auto prepared = preprocess(region);
    near(prepared.input[0], 114.0F / 255.0F);
    near(prepared.input[340], 114.0F / 255.0F);
    near(prepared.input[341], 1.0F);
    near(prepared.input[681], 1.0F);
    near(prepared.input[682], 114.0F / 255.0F);
    require(prepared.bgr.isContinuous(), "diagnostic decoded image must be contiguous");
}

void invalid_preprocessing_and_input() {
    rejects([] { preprocess(cv::Mat{}); });
    rejects([] { preprocess(cv::Mat(2, 2, CV_8UC1)); });
    rejects([] { preprocess(cv::Mat(2, 2, CV_8UC4)); });
    rejects([] { preprocess(cv::Mat(2, 2, CV_16UC3)); });
    rejects([] { validate_input({}); });
    std::vector<float> input(input_elements, 0.0F);
    for (const float invalid : {nan_value, infinity, -0.001F, 1.001F}) {
        input[4] = invalid;
        rejects([&] { validate_input(input); });
    }
}

void lossless_decode_and_rejections() {
    TemporaryDirectory temporary;
    const cv::Mat bgr(3, 5, CV_8UC3, cv::Scalar(13, 27, 249));
    for (const auto* extension : {".png", ".bmp", ".PNG"}) {
        const auto path = (temporary.path / ("image" + std::string(extension))).string();
        require(cv::imwrite(path, bgr), "write lossless fixture");
        const auto restored = decode_image(path);
        require(restored.size() == bgr.size() &&
                cv::norm(restored, bgr, cv::NORM_INF) == 0.0, "exact lossless pixels");
    }
    for (const int type : {CV_8UC1, CV_8UC4, CV_16UC3}) {
        const auto path = (temporary.path / ("unsupported-" + std::to_string(type) + ".png")).string();
        require(cv::imwrite(path, cv::Mat(3, 5, type, cv::Scalar::all(0))), "write invalid fixture");
        rejects([&] { decode_image(path); });
    }
    rejects([&] { decode_image((temporary.path / "missing.png").string()); });
    rejects([&] { decode_image((temporary.path / "image.jpg").string()); });
    const auto jpeg = (temporary.path / "masqueraded.jpg").string();
    require(cv::imwrite(jpeg, bgr), "write JPEG fixture");
    const auto mislabeled = (temporary.path / "masqueraded.png").string();
    std::filesystem::copy_file(jpeg, mislabeled);
    rejects([&] { decode_image(mislabeled); });
}

void decode_single_label_and_confidence() {
    const auto raw = output({{100, 80, 20, 10, 0.5F, 0.5F},
                             {200, 80, 20, 10, 0.25F, 0.25F},
                             {300, 80, 20, 10, 0.1F, std::nextafter(0.25F, 1.0F)}});
    const auto detections = postprocess(raw, letterbox_geometry(1024, 1024));
    require(detections.size() == 2, "strict score > confidence");
    require(detections[0].class_id == 0 && detections[0].candidate_index == 0, "argmax tie class zero");
    require(detections[1].class_id == 1 && detections[1].candidate_index == 2, "single label");
    require(detections[0].xyxy == std::array<float, 4>{90, 75, 110, 85}, "cxcywh decoded once");
    near(detections[0].score, 0.5F);
}

void class_aware_nms_and_score_ties() {
    const auto raw = output({{100, 100, 20, 20, 0.9F, 0.1F},
                             {100, 100, 20, 20, 0.9F, 0.1F},
                             {100, 100, 20, 20, 0.1F, 0.9F},
                             {200, 100, 20, 20, 0.9F, 0.1F}});
    const auto detections = postprocess(raw, letterbox_geometry(1024, 1024));
    require(detections.size() == 3 && detections[0].candidate_index == 0 &&
            detections[1].candidate_index == 2 && detections[2].candidate_index == 3,
            "class-aware NMS and ascending candidate-index tie order");
}

void iou_strict_boundary() {
    // Nested boxes: intersection 2, union 4 -> exact IoU 0.5.
    const auto raw = output({{5, 5, 2, 2, 0.9F, 0.0F}, {5, 5, 1, 2, 0.8F, 0.0F}});
    const auto geometry = letterbox_geometry(1024, 1024);
    require(postprocess(raw, geometry, 0.25F, 0.5F).size() == 2, "IoU equality is retained");
    require(postprocess(raw, geometry, 0.25F, std::nextafter(0.5F, 0.0F)).size() == 1,
            "IoU strictly above threshold is suppressed");
    require(postprocess(raw, geometry, 0.25F, std::nextafter(0.5F, 1.0F)).size() == 2,
            "IoU below threshold is retained");
    const auto deployment_boundary = output({{10, 10, 10, 1, 0.9F, 0},
                                               {10, 10, 7, 1, 0.8F, 0}});
    require(postprocess(deployment_boundary, geometry).size() == 2,
            "actual deployment IoU=0.7 equality is retained");
    require(postprocess(deployment_boundary, geometry, 0.25F,
                        std::nextafter(0.7F, 0.0F)).size() == 1,
            "actual deployment IoU threshold neighbor");
    const auto identical = output({{5, 5, 2, 2, 0.9F, 0}, {5, 5, 2, 2, 0.8F, 0}});
    require(postprocess(identical, geometry, 0.25F, 1.0F).size() == 2, "IoU one equality retained");
}

void inverse_coordinates_and_clip() {
    const auto geometry = letterbox_geometry(2048, 1024);
    const auto raw = output({{100, 356, 50, 40, 0.9F, 0.0F},
                             {0, 256, 10, 10, 0.8F, 0.0F},
                             {1024, 768, 10, 10, 0.7F, 0.0F}});
    const auto detections = postprocess(raw, geometry);
    require(detections[0].xyxy == std::array<float, 4>{150, 160, 250, 240}, "inverse exact");
    require(detections[1].xyxy == std::array<float, 4>{0, 0, 10, 10}, "clip lower edges");
    require(detections[2].xyxy == std::array<float, 4>{2038, 1014, 2048, 1024}, "clip upper edges");
    const auto odd = letterbox_geometry(333, 1000);
    const auto boxes = postprocess(output({{442, 200, 1, 1, 0.9F, 0}}), odd);
    near(boxes[0].xyxy[0], (441.5F - 341.0F) / 1.024F);
    require(boxes[0].xyxy[0] != std::floor(boxes[0].xyxy[0]), "floating coordinates preserved");
}

void nms_before_restoration_and_clipping() {
    const auto raw = output({{-10, 5, 20, 10, 0.9F, 0}, {0, 5, 20, 10, 0.8F, 0}});
    // Before clipping IoU=1/3; after clipping first box has zero area.
    require(postprocess(raw, letterbox_geometry(1024, 1024), 0.25F, 0.3F).size() == 1,
            "NMS must precede clipping/restoration");
}

void empty_and_max_det() {
    const auto geometry = letterbox_geometry(1024, 1024);
    require(postprocess(output({{5, 5, 2, 2, 0, 0}}), geometry).empty(), "empty valid result");
    const auto raw = output({{10, 10, 2, 2, 0.5F, 0}, {20, 10, 2, 2, 0.9F, 0},
                             {30, 10, 2, 2, 0.7F, 0}});
    const auto kept = postprocess(raw, geometry, 0.25F, 0.7F, 2);
    require(kept.size() == 2 && kept[0].candidate_index == 1 && kept[1].candidate_index == 2,
            "score ordering and max_det");
    RawOutput many{std::vector<float>(6 * 301, 0.0F), 301};
    for (std::size_t i = 0; i < many.candidates; ++i) {
        many.values[i] = 2.0F * static_cast<float>(i) + 1.0F;
        many.values[many.candidates + i] = 10.0F;
        many.values[2 * many.candidates + i] = 1.0F;
        many.values[3 * many.candidates + i] = 1.0F;
        many.values[4 * many.candidates + i] = 0.9F;
    }
    const auto deployment = postprocess(many, geometry);
    require(deployment.size() == 300 && deployment.back().candidate_index == 299,
            "actual max_det=300 with deterministic equal-score order");
}

void invalid_outputs() {
    const auto geometry = letterbox_geometry(1024, 1024);
    rejects([&] { postprocess({{}, 0}, geometry); });
    rejects([&] { postprocess({{1, 2}, 1}, geometry); });
    for (std::size_t channel = 0; channel < 6; ++channel) {
        auto raw = output({{10, 10, 2, 2, 0.5F, 0}});
        raw.values[channel] = nan_value;
        rejects([&] { postprocess(raw, geometry); });
        raw.values[channel] = infinity;
        rejects([&] { postprocess(raw, geometry); });
    }
    for (const auto& raw : {output({{5, 5, -1, 2, 0.5F, 0}}),
                           output({{5, 5, 1, -2, 0.5F, 0}}),
                           output({{5, 5, 1, 2, -0.1F, 0}}),
                           output({{5, 5, 1, 2, 1.1F, 0}}),
                           output({{5, 5, 1e30F, 1e30F, 0.5F, 0}})}) {
        rejects([&] { postprocess(raw, geometry); });
    }
    const auto valid = output({{5, 5, 2, 2, 0.5F, 0}});
    rejects([&] { postprocess(valid, geometry, nan_value); });
    rejects([&] { postprocess(valid, geometry, 0.25F, -0.1F); });
    rejects([&] { postprocess(valid, geometry, 0.25F, 0.7F, 0); });
    auto bad_geometry = geometry;
    bad_geometry.left = 1;
    rejects([&] { postprocess(valid, bad_geometry); });
}

void serialization_and_binary_io() {
    const auto geometry = letterbox_geometry(1024, 1024);
    const auto empty = detection_json({}, geometry);
    require(empty["schema_version"] == 1 && empty["detections"].empty() &&
            empty["classes"]["0"] == "crop" && empty["classes"]["1"] == "weed", "empty JSON schema");
    auto boxes = postprocess(output({{5, 5, 2, 2, 0.1F, 0.9F}}), geometry);
    const auto encoded = detection_json(boxes, geometry);
    require(encoded["detections"][0]["class_name"] == "weed" &&
            encoded["detections"][0]["candidate_index"] == 0, "JSON class/index");
    require(encoded.dump() == detection_json(boxes, geometry).dump(), "deterministic serialization");
    TemporaryDirectory temporary;
    const auto json_path = (temporary.path / "result.json").string();
    write_json(json_path, encoded);
    std::ifstream stream(json_path);
    nlohmann::json read;
    stream >> read;
    require(read == encoded, "JSON round trip");
    boxes[0].score = nan_value;
    rejects([&] { detection_json(boxes, geometry); });
    boxes[0].score = 0.5F;
    boxes[0].class_id = 2;
    rejects([&] { detection_json(boxes, geometry); });
    boxes[0].class_id = 1;
    boxes[0].xyxy[0] = infinity;
    rejects([&] { detection_json(boxes, geometry); });
    boxes[0].xyxy = {3, 4, 2, 5};
    rejects([&] { detection_json(boxes, geometry); });
    rejects([&] { write_json((temporary.path / "missing" / "x.json").string(), encoded); });
    const auto tensor_path = (temporary.path / "input.f32").string();
    std::vector<float> input(input_elements, 0.5F);
    write_binary(tensor_path, input.data(), input.size() * sizeof(float));
    require(read_input_tensor(tensor_path) == input, "exact input tensor round trip");
    write_binary(tensor_path, input.data(), 1);
    rejects([&] { read_input_tensor(tensor_path); });
}

void invalid_model_files() {
    TemporaryDirectory temporary;
    const auto missing = (temporary.path / "missing.onnx").string();
    rejects([&] { Session model(missing); });
    const auto malformed = (temporary.path / "malformed.onnx").string();
    const std::string garbage = "This is not an ONNX graph.";
    write_binary(malformed, garbage.data(), garbage.size());
    rejects([&] { Session model(malformed); });
}

int real_model(int argc, char** argv) {
    if (argc != 4 || !std::filesystem::is_regular_file(argv[2]) ||
        !std::filesystem::is_regular_file(argv[3])) {
        std::cout << "NOT RUN: provide SMARTSPRAY_REAL_MODEL and SMARTSPRAY_REAL_IMAGE artifacts\n";
        return 77;
    }
    Session session(argv[2]);
    const auto image = prepare_image(argv[3]);
    const auto first = session.infer(image.input);
    const auto second = session.infer(image.input);
    require(first.values == second.values, "reused CPU session must be deterministic");
    const auto result = detection_json(postprocess(first, image.geometry), image.geometry);
    require(result["schema_version"] == 1, "real-model serialization");
    std::cout << "PASS real_model: " << session.candidates() << " candidates, "
              << result["detections"].size() << " detections\n";
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        cv::setNumThreads(1);
        if (argc > 1 && std::string(argv[1]) == "--real") return real_model(argc, argv);
        const std::vector<std::pair<const char*, std::function<void()>>> tests{
            {"rounding_ties", rounding_ties}, {"geometry_odd_and_ties", geometry_odd_and_ties},
            {"rgb_normalization_and_layout", rgb_normalization_and_layout},
            {"padding_and_noncontiguous_image", padding_and_noncontiguous_image},
            {"invalid_preprocessing_and_input", invalid_preprocessing_and_input},
            {"lossless_decode_and_rejections", lossless_decode_and_rejections},
            {"decode_single_label_and_confidence", decode_single_label_and_confidence},
            {"class_aware_nms_and_score_ties", class_aware_nms_and_score_ties},
            {"iou_strict_boundary", iou_strict_boundary},
            {"inverse_coordinates_and_clip", inverse_coordinates_and_clip},
            {"nms_before_restoration_and_clipping", nms_before_restoration_and_clipping},
            {"empty_and_max_det", empty_and_max_det}, {"invalid_outputs", invalid_outputs},
            {"serialization_and_binary_io", serialization_and_binary_io},
            {"invalid_model_files", invalid_model_files}};
        for (const auto& [name, test] : tests) {
            test();
            std::cout << "PASS " << name << "\n";
        }
        std::cout << tests.size() << " offline vision scenarios passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "vision_tests: " << error.what() << "\n";
        return 1;
    }
}

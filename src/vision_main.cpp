#include "vision_runtime.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>

namespace {
using namespace smartspray::vision;
using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

nlohmann::json statistics(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const auto percentile = [&values](double quantile) {
        const double index = quantile * static_cast<double>(values.size() - 1);
        const auto low = static_cast<std::size_t>(index);
        const auto high = std::min(low + 1, values.size() - 1);
        return values[low] + (values[high] - values[low]) * (index - low);
    };
    return {{"median_ms", percentile(0.5)}, {"p95_ms", percentile(0.95)}};
}

void check_paths(const std::map<std::string, std::string>& args) {
    std::set<std::filesystem::path> input_paths;
    for (const char* key : {"--model", "--image", "--tensor"}) {
        const auto found = args.find(key);
        if (found != args.end()) input_paths.insert(std::filesystem::weakly_canonical(found->second));
    }
    std::vector<std::string> outputs{args.at("--output")};
    if (args.count("--timing-output")) outputs.push_back(args.at("--timing-output"));
    if (args.count("--dump-prefix")) {
        for (const char* suffix : {".bgr.u8", ".input.f32", ".raw.f32", ".geometry.json"}) {
            outputs.push_back(args.at("--dump-prefix") + suffix);
        }
    }
    std::set<std::filesystem::path> output_paths;
    for (const auto& output : outputs) {
        const auto path = std::filesystem::weakly_canonical(output);
        const auto aliases = [&path](const auto& paths) {
            for (const auto& other : paths) {
                if (path == other || (std::filesystem::exists(path) &&
                    std::filesystem::exists(other) && std::filesystem::equivalent(path, other))) {
                    return true;
                }
            }
            return false;
        };
        if (aliases(input_paths) || aliases(output_paths)) {
            throw std::runtime_error("output paths must be distinct from each other and from input files");
        }
        output_paths.insert(path);
    }
}
}  // namespace

int main(int argc, char** argv) {
    try {
        std::map<std::string, std::string> args;
        const std::set<std::string> options{"--model", "--image", "--output", "--dump-prefix",
                                             "--tensor", "--benchmark", "--timing-output"};
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "smartspray_infer --model MODEL.onnx --image IMAGE.png --output RESULT.json\n"
                         "  [--dump-prefix PATH] [--tensor INPUT.f32]\n"
                         "  [--benchmark N --timing-output TIMING.json]\n"
                         "PNG/BMP: three-channel 8-bit only; no EXIF rotation.\n"
                         "--tensor replaces preprocessed input for diagnostics; image geometry still comes from decoding.\n"
                         "--benchmark needs N >= 30, uses 5 warmups and one CPU session.\n";
            return 0;
        }
        for (int index = 1; index < argc; index += 2) {
            const std::string key = argv[index];
            if (!options.count(key) || index + 1 >= argc || !args.emplace(key, argv[index + 1]).second) {
                throw std::runtime_error("unknown, duplicate or missing CLI option; use --help");
            }
        }
        for (const char* required : {"--model", "--image", "--output"}) {
            if (!args.count(required) || args.at(required).empty()) {
                throw std::runtime_error(std::string("missing required argument: ") + required);
            }
        }
        int benchmark_count = 0;
        if (args.count("--benchmark")) {
            std::size_t consumed = 0;
            benchmark_count = std::stoi(args.at("--benchmark"), &consumed);
            if (consumed != args.at("--benchmark").size() || benchmark_count < 30) {
                throw std::runtime_error("--benchmark requires an integer >= 30");
            }
            if (args.count("--tensor")) throw std::runtime_error("benchmark requires native image preprocessing");
            if (!args.count("--timing-output")) args["--timing-output"] = args.at("--output") + ".timing.json";
        } else if (args.count("--timing-output")) {
            throw std::runtime_error("--timing-output requires --benchmark");
        }
        check_paths(args);
        cv::setNumThreads(1);
        const auto load_start = Clock::now();
        Session session(args.at("--model"));
        const double model_load_ms = elapsed_ms(load_start, Clock::now());
        auto prepared = prepare_image(args.at("--image"));
        if (args.count("--tensor")) prepared.input = read_input_tensor(args.at("--tensor"));
        auto raw = session.infer(prepared.input);
        auto detections = postprocess(raw, prepared.geometry);
        write_json(args.at("--output"), detection_json(detections, prepared.geometry));
        if (args.count("--dump-prefix")) {
            const auto prefix = args.at("--dump-prefix");
            write_binary(prefix + ".bgr.u8", prepared.bgr.data, prepared.bgr.total() * prepared.bgr.elemSize());
            write_binary(prefix + ".input.f32", prepared.input.data(), prepared.input.size() * sizeof(float));
            write_binary(prefix + ".raw.f32", raw.values.data(), raw.values.size() * sizeof(float));
            auto geometry = geometry_json(prepared.geometry);
            geometry["input_shape"] = {1, 3, input_size, input_size};
            geometry["output_shape"] = {1, 6, raw.candidates};
            geometry["input_name"] = session.input_name();
            geometry["output_name"] = session.output_name();
            geometry["diagnostic_tensor_override"] = args.count("--tensor") != 0;
            write_json(prefix + ".geometry.json", geometry);
        }
        if (benchmark_count) {
            std::vector<double> preprocessing_times, inference_times, postprocessing_times;
            for (int run = -5; run < benchmark_count; ++run) {
                const auto start = Clock::now();
                auto image = prepare_image(args.at("--image"));
                const auto preprocessed = Clock::now();
                auto prediction = session.infer(image.input);
                const auto inferred = Clock::now();
                const auto boxes = postprocess(prediction, image.geometry);
                const auto postprocessed = Clock::now();
                if (run >= 0) {
                    preprocessing_times.push_back(elapsed_ms(start, preprocessed));
                    inference_times.push_back(elapsed_ms(preprocessed, inferred));
                    postprocessing_times.push_back(elapsed_ms(inferred, postprocessed));
                }
            }
            write_json(args.at("--timing-output"),
                       {{"schema_version", 1}, {"runtime", Session::runtime_version()},
                        {"provider", "CPUExecutionProvider"}, {"execution_mode", "sequential"},
                        {"intra_op_threads", 1}, {"inter_op_threads", 1}, {"opencv_threads", 1},
                        {"graph_optimization", "ORT_ENABLE_ALL"}, {"warmup_runs", 5},
                        {"measured_runs", benchmark_count}, {"model_load_ms", model_load_ms},
                        {"preprocessing", statistics(preprocessing_times)},
                        {"inference", statistics(inference_times)},
                        {"postprocessing", statistics(postprocessing_times)}});
        }
        std::cout << "detections=" << detections.size() << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "smartspray_infer: " << error.what() << "\n";
        return 1;
    }
}

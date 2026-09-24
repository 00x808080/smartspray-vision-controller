#include "demo.hpp"
#include "sha256.hpp"
#include "vision_runtime.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        using namespace smartspray;
        using Clock=std::chrono::steady_clock;
        const auto start=Clock::now();
        std::map<std::string,std::string> args;
        const std::set<std::string> options{"--model","--image","--config","--output"};
        if (argc==2 && std::string(argv[1])=="--help") {
            std::cout << "smartspray_vision_demo --model MODEL.onnx --image IMAGE.png --config CONFIG.json --output NEW_DIRECTORY\n"
                         "Native CPU inference and eight-channel virtual commands. Output parent must exist.\n";
            return 0;
        }
        for (int i=1; i<argc; i+=2) {
            if (!options.count(argv[i]) || i+1>=argc || !args.emplace(argv[i],argv[i+1]).second)
                throw std::runtime_error("Unknown, duplicate or missing CLI argument; use --help");
        }
        for (const auto& name:options)
            if (!args.count(name) || args.at(name).empty()) throw std::runtime_error("Missing argument: "+name);
        std::ifstream config_file(args.at("--config"));
        if (!config_file) throw std::runtime_error("Cannot read configuration");
        const auto config=nlohmann::json::parse(config_file); // Strict whole-document parsing.
        const auto profile=simulation::parse_profile(config); // Validate before pipeline.
        demo::require_new_output(args.at("--output"));
        const auto model_hash=sha256_file(args.at("--model"));
        const auto image_hash=sha256_file(args.at("--image"));
        cv::setNumThreads(1);
        vision::Session session(args.at("--model"));
        const auto loaded=Clock::now();
        auto image=vision::prepare_image(args.at("--image"));
        const auto prepared=Clock::now();
        auto raw=session.infer(image.input);
        const auto inferred=Clock::now();
        auto detections=vision::postprocess(raw,image.geometry);
        auto run=demo::compose(detections,image.geometry,profile);
        run["image"]={{"id",std::filesystem::path(args.at("--image")).stem().string()},
                      {"filename",std::filesystem::path(args.at("--image")).filename().string()},
                      {"sha256",image_hash},{"original_width_px",image.bgr.cols},
                      {"original_height_px",image.bgr.rows}};
        run["model"]={{"sha256",model_hash},{"input_name",session.input_name()},
                      {"output_name",session.output_name()},{"dtype","FP32"},
                      {"input_shape",{1,3,vision::input_size,vision::input_size}},
                      {"output_shape",{1,6,session.candidates()}}};
        run["runtime"]={{"name","ONNX Runtime"},{"version",vision::Session::runtime_version()},
            {"provider","CPUExecutionProvider"},{"execution_mode","sequential"},
            {"intra_op_threads",1},{"inter_op_threads",1},{"opencv_threads",1},
            {"graph_optimization","ORT_ENABLE_ALL"},{"opencv_version",CV_VERSION}};
        auto annotation=demo::render_annotated(image.bgr,run);
        auto timeline=demo::render_timeline(run);
        if (model_hash!=sha256_file(args.at("--model")) || image_hash!=sha256_file(args.at("--image")))
            throw std::runtime_error("Model/image changed during run");
        demo::write_outputs(args.at("--output"),run,annotation,timeline);
        const auto elapsed=[](auto a,auto b){return std::chrono::duration<double,std::milli>(b-a).count();};
        // Nondeterministic measurements and output path are console metadata only.
        std::cout << nlohmann::json{{"output_directory",args.at("--output")},
            {"counts",run.at("counts")},{"all_channels_off",true},
            {"wall_clock_ms",{{"setup_and_load",elapsed(start,loaded)},
                             {"decode_preprocess",elapsed(loaded,prepared)},
                             {"inference",elapsed(prepared,inferred)},
                             {"total",elapsed(start,Clock::now())}}}}.dump(2) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "smartspray_vision_demo: " << error.what() << '\n';
        return 1;
    }
}

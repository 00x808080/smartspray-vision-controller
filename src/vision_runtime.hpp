#pragma once

#include "vision.hpp"
#include <onnxruntime_cxx_api.h>
#include <string>

namespace smartspray::vision {

class Session {
public:
    explicit Session(const std::string& model_path);
    RawOutput infer(const std::vector<float>& input);
    std::size_t candidates() const { return candidates_; }
    const std::string& input_name() const { return input_name_; }
    const std::string& output_name() const { return output_name_; }
    static const char* runtime_version();

private:
    Ort::Env environment_{ORT_LOGGING_LEVEL_WARNING, "smartspray"};
    Ort::SessionOptions options_;
    Ort::Session session_{nullptr};
    std::string input_name_;
    std::string output_name_;
    std::size_t candidates_ = 0;
};

}  // namespace smartspray::vision

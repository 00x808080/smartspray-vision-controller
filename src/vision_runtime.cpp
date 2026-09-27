#include "vision_runtime.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace smartspray::vision {

Session::Session(const std::string& model_path) {
    options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    options_.SetIntraOpNumThreads(1);
    options_.SetInterOpNumThreads(1);
    options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    // No execution provider is appended: the official CPU runtime uses CPU EP.
    session_ = Ort::Session(environment_, model_path.c_str(), options_);
    if (session_.GetInputCount() != 1 || session_.GetOutputCount() != 1) {
        throw std::runtime_error("model must have exactly one input and one output");
    }
    const auto input_info = session_.GetInputTypeInfo(0);
    const auto output_info = session_.GetOutputTypeInfo(0);
    if (input_info.GetONNXType() != ONNX_TYPE_TENSOR ||
        output_info.GetONNXType() != ONNX_TYPE_TENSOR) {
        throw std::runtime_error("model input/output must be tensors");
    }
    const auto input_tensor = input_info.GetTensorTypeAndShapeInfo();
    const auto output_tensor = output_info.GetTensorTypeAndShapeInfo();
    const auto output_shape = output_tensor.GetShape();
    if (input_tensor.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        input_tensor.GetShape() != std::vector<int64_t>({1, 3, input_size, input_size}) ||
        output_tensor.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        output_shape.size() != 3 || output_shape[0] != 1 || output_shape[1] != 6 ||
        output_shape[2] <= 0 ||
        static_cast<std::uint64_t>(output_shape[2]) >
            std::numeric_limits<std::size_t>::max() / 6) {
        throw std::runtime_error("model requires FP32 input [1,3,1024,1024] and output [1,6,N], N > 0");
    }
    candidates_ = static_cast<std::size_t>(output_shape[2]);
    Ort::AllocatorWithDefaultOptions allocator;
    input_name_ = session_.GetInputNameAllocated(0, allocator).get();
    output_name_ = session_.GetOutputNameAllocated(0, allocator).get();
    const auto metadata = session_.GetModelMetadata();
    const auto names_value = metadata.LookupCustomMetadataMapAllocated("names", allocator);
    const auto task_value = metadata.LookupCustomMetadataMapAllocated("task", allocator);
    if (!names_value || !task_value || std::string(task_value.get()) != "detect") {
        throw std::runtime_error("model lacks the required detector task/class metadata");
    }
    const auto end2end = metadata.LookupCustomMetadataMapAllocated("end2end", allocator);
    if (end2end && std::string(end2end.get()) != "False") {
        throw std::runtime_error("end-to-end/NMS output is unsupported");
    }
    std::string names(names_value.get());
    names.erase(std::remove_if(names.begin(), names.end(),
                              [](unsigned char ch) { return std::isspace(ch); }), names.end());
    if (names != "{0:'crop',1:'weed'}" && names != "{0:\"crop\",1:\"weed\"}" &&
        names != "{\"0\":\"crop\",\"1\":\"weed\"}") {
        throw std::runtime_error("model metadata must declare exactly 0=crop and 1=weed");
    }
}

RawOutput Session::infer(const std::vector<float>& input) {
    validate_input(input);
    const std::array<int64_t, 4> shape{1, 3, input_size, input_size};
    const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<float>(
        memory, const_cast<float*>(input.data()), input.size(), shape.data(), shape.size());
    const char* input_names[] = {input_name_.c_str()};
    const char* output_names[] = {output_name_.c_str()};
    auto outputs = session_.Run(Ort::RunOptions{nullptr}, input_names, &tensor, 1, output_names, 1);
    if (outputs.size() != 1 || !outputs[0].IsTensor()) {
        throw std::runtime_error("inference returned an invalid output");
    }
    const auto information = outputs[0].GetTensorTypeAndShapeInfo();
    if (information.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        information.GetShape() != std::vector<int64_t>({1, 6, static_cast<int64_t>(candidates_)})) {
        throw std::runtime_error("inference output shape/type changed");
    }
    const float* values = outputs[0].GetTensorData<float>();
    RawOutput result{{values, values + 6 * candidates_}, candidates_};
    for (float value : result.values) {
        if (!std::isfinite(value)) throw std::runtime_error("inference produced non-finite output");
    }
    return result;
}

const char* Session::runtime_version() { return OrtGetApiBase()->GetVersionString(); }

}  // namespace smartspray::vision

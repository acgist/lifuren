#include "lifuren/Torch.hpp"

#include "torch/torch.h"

#include "spdlog/spdlog.h"
#include "spdlog/fmt/ostr.h"

#include "lifuren/Logger.hpp"

LFR_FORMAT_LOG_STREAM(at::Tensor)
LFR_FORMAT_LOG_STREAM(c10::IntArrayRef)

void lifuren::setDevice(torch::DeviceType& type) {
    if(torch::cuda::is_available()) {
        type = torch::DeviceType::CUDA;
    } else {
        type = torch::DeviceType::CPU;
    }
}

void lifuren::logTensor(const torch::Tensor& tensor) {
    SPDLOG_DEBUG("{}", tensor);
}

void lifuren::logTensor(const c10::IntArrayRef& tensor) {
    SPDLOG_DEBUG("{}", tensor);
}

void lifuren::quantization(const std::string& model_path) {
    // TODO: 实现
}

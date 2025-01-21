#include "lifuren/Torch.hpp"

#include "torch/cuda.h"
#include "torch/data.h"

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

void lifuren::logTensor(const std::string& message, const at::Tensor& tensor) {
    SPDLOG_DEBUG("{}\n{}", message, tensor);
}

void lifuren::logTensor(const std::string& message, const c10::IntArrayRef& array) {
    SPDLOG_DEBUG("{}\n{}", message, array);
}

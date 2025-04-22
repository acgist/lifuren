#include "lifuren/Torch.hpp"

#include "torch/cuda.h"
#include "torch/data.h"

#include "spdlog/spdlog.h"
#include "spdlog/fmt/ostr.h"

#include "lifuren/Logger.hpp"

#if FMT_VERSION > 100000
LFR_FORMAT_LOG_STREAM(at::Tensor)
LFR_FORMAT_LOG_STREAM(c10::IntArrayRef)
#endif

torch::DeviceType lifuren::getDevice() {
    if(torch::cuda::is_available()) {
        return torch::DeviceType::CUDA;
    } else {
        return torch::DeviceType::CPU;
    }
}

void lifuren::logTensor(const std::string& message, const at::Tensor& tensor) {
    SPDLOG_DEBUG("{}\n{}", message, tensor);
}

void lifuren::logTensor(const std::string& message, const c10::IntArrayRef& tensor) {
    SPDLOG_DEBUG("{}\n{}", message, tensor);
}

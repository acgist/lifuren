#include "../../header/LibTorch.hpp"

void lifuren::testCUDA() {
    SPDLOG_DEBUG("CUDA = {}", torch::cuda::is_available());
    SPDLOG_DEBUG("CUDA count = {}", torch::cuda::device_count());
    SPDLOG_DEBUG("CUDA cudnn = {}", torch::cuda::cudnn_is_available());
}

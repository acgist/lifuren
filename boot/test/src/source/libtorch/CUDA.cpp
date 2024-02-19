#include "../../header/CUDA.hpp"

void lifuren::testCUDA() {
    LOG(INFO) << "CUDA = "       << torch::cuda::is_available();
    LOG(INFO) << "CUDA count = " << torch::cuda::device_count();
    LOG(INFO) << "CUDA cudnn = " << torch::cuda::cudnn_is_available();
}
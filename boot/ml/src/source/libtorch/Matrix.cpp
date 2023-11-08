#include "../../header/LibTorch.hpp"

void lifuren::testMatrix() {
    const torch::Tensor a = torch::randn({ 3, 2 });
    const torch::Tensor b = torch::randn({ 3, 2 });
    LOG(INFO) << a;
    LOG(INFO) << b;
    LOG(INFO) << (a + b);
    // torch::Tensor output = torch::randn({ 3, 2 });
    // LOG(INFO) << output;
    // torch::kCPU;
    // torch::kCUDA;
    // LOG(INFO) << "是否支持CUDA："  << torch::cuda::is_available();
    // LOG(INFO) << "是否支持CUDNN：" << torch::cuda::cudnn_is_available();
}

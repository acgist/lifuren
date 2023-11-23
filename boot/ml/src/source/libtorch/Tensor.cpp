#include "../../header/LibTorch.hpp"

void lifuren::testLibTorchTensor() {
    const torch::Tensor a = torch::randn({ 3, 2 });
    const torch::Tensor b = torch::randn({ 3, 2 });
    LOG(INFO) << "a =" << std::endl << a;
    LOG(INFO) << "b =" << std::endl << b;
    LOG(INFO) << "a + b =" << std::endl << (a + b);
    LOG(INFO) << "是否支持CUDA：" << torch::cuda::is_available();
    LOG(INFO) << "是否支持CUDNN：" << torch::cuda::cudnn_is_available();
}

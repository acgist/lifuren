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

void lifuren::testReLU() {
    torch::nn::ReLU relu;
    LOG(INFO) << relu;
    at::Tensor input  = torch::randint(0, 10, {1, 1, 4});
    at::Tensor output = relu->forward(input);
    // relu(input); // TODO：测试
    LOG(INFO) << input;
    LOG(INFO) << output;
}

void lifuren::testTanh() {
    torch::nn::Tanh tanh;
    LOG(INFO) << tanh;
    at::Tensor tensor;
    at::Tensor input  = torch::randint(0, 10, {1, 1, 4});
    at::Tensor output = tanh->forward(input);
    LOG(INFO) << input;
    LOG(INFO) << output;
}
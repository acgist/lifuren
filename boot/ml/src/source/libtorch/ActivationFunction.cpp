#include "../../header/LibTorch.hpp"

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

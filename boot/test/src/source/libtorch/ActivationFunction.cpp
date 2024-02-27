#include "../../header/LibTorch.hpp"

template <> struct fmt::formatter<torch::nn::ReLU> : ostream_formatter {};
template <> struct fmt::formatter<torch::nn::Tanh> : ostream_formatter {};

void lifuren::testLibTorchReLU() {
    torch::nn::ReLU relu;
    SPDLOG_DEBUG("RELU：{}", relu);
    at::Tensor input  = torch::randint(0, 10, { 1, 1, 4 });
    at::Tensor output = relu->forward(input);
    SPDLOG_DEBUG("input： {}", input);
    SPDLOG_DEBUG("output：{}", output);
}

void lifuren::testLibTorchTanh() {
    torch::nn::Tanh tanh;
    SPDLOG_DEBUG("tanh：{}", tanh);
    at::Tensor tensor;
    at::Tensor input  = torch::randint(0, 10, { 1, 1, 4 });
    at::Tensor output = tanh->forward(input);
    SPDLOG_DEBUG("input： {}", input);
    SPDLOG_DEBUG("output：{}", output);
}

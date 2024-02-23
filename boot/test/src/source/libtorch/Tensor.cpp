#include "../../header/LibTorch.hpp"

#include "Format.hpp"

void lifuren::testLibTorchTensor() {
    // const torch::Tensor a = torch::randn({ 3, 2 });
    // const torch::Tensor b = torch::randn({ 3, 2 });
    int arrayA[] = { 1, 2, 3, 4 };
    int arrayB[] = { 1, 2, 3, 4 };
    const torch::Tensor lineA = torch::tensor({ 1, 2, 3, 4});
    const torch::Tensor lineB = torch::tensor({ 1, 2, 3, 4});
    const torch::Tensor a = torch::from_blob(arrayA, { 2, 2 }, torch::kInt);
    const torch::Tensor b = torch::from_blob(arrayB, { 2, 2 }, torch::kInt);
    SPDLOG_DEBUG("a =\r\n{}", a);
    SPDLOG_DEBUG("b =\r\n{}", b);
    SPDLOG_DEBUG("a + b =\r\n{}", (a + b));
    SPDLOG_DEBUG("a - b =\r\n{}", (a - b));
    SPDLOG_DEBUG("a * b =\r\n{}", (a * b));
    SPDLOG_DEBUG("a / b =\r\n{}", (a / b));
    SPDLOG_DEBUG("a % b =\r\n{}", (a % b));
    SPDLOG_DEBUG("a == b =\r\n{}", (a == b));
    SPDLOG_DEBUG("lineA dot lineB =\r\n{}", lineA.dot(lineB));
    SPDLOG_DEBUG("lineA dot lineB.t =\r\n{}", lineA.dot(lineB.t()));
    SPDLOG_DEBUG("是否支持CUDA：{}", torch::cuda::is_available());
    SPDLOG_DEBUG("是否支持CUDNN：{}", torch::cuda::cudnn_is_available());
}

#include "../../header/LibTorch.hpp"

void lifuren::testLibTorchTensor() {
    // const torch::Tensor a = torch::randn({ 3, 2 });
    // const torch::Tensor b = torch::randn({ 3, 2 });
    int arrayA[] = { 1, 2, 3, 4 };
    int arrayB[] = { 1, 2, 3, 4 };
    const torch::Tensor lineA = torch::tensor({ 1, 2, 3, 4});
    const torch::Tensor lineB = torch::tensor({ 1, 2, 3, 4});
    const torch::Tensor a = torch::from_blob(arrayA, { 2, 2 }, torch::kInt);
    const torch::Tensor b = torch::from_blob(arrayB, { 2, 2 }, torch::kInt);
    LOG(INFO) << "a =" << std::endl << a;
    LOG(INFO) << "b =" << std::endl << b;
    LOG(INFO) << "a + b =" << std::endl << (a + b);
    LOG(INFO) << "a - b =" << std::endl << (a - b);
    LOG(INFO) << "a * b =" << std::endl << (a * b);
    LOG(INFO) << "a / b =" << std::endl << (a / b);
    LOG(INFO) << "a % b =" << std::endl << (a % b);
    LOG(INFO) << "a == b =" << std::endl << (a == b);
    LOG(INFO) << "lineA dot lineB =" << std::endl << lineA.dot(lineB);
    LOG(INFO) << "lineA dot lineB.t =" << std::endl << lineA.dot(lineB.t());
    // LOG(INFO) << "a + b =" << std::endl << torch::pixel_shuffle(a);
    // LOG(INFO) << "a + b =" << std::endl << torch::channel_shuffle(a);
    LOG(INFO) << "是否支持CUDA：" << torch::cuda::is_available();
    LOG(INFO) << "是否支持CUDNN：" << torch::cuda::cudnn_is_available();
}

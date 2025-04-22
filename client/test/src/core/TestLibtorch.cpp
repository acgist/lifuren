#include "lifuren/Test.hpp"

#include <fstream>

#include "torch/torch.h"
#include "torch/script.h"

#include "spdlog/spdlog.h"

#include "lifuren/File.hpp"
#include "lifuren/Layer.hpp"
#include "lifuren/Torch.hpp"
#include "lifuren/Config.hpp"

[[maybe_unused]] static void testJit() {
    auto model = torch::jit::load(lifuren::file::join({ lifuren::config::CONFIG.tmp, "lifuren.pth" }).string());
    std::vector<torch::jit::IValue> inputs;
    auto input = torch::randn({ 1 });
    inputs.push_back(std::move(input));
    model.eval();
    auto tensor = model.forward(inputs);
    auto result = tensor.toTensor().template item<float>();
    lifuren::logTensor("result", result);
}

[[maybe_unused]] static void testCuda() {
    SPDLOG_DEBUG("cuda：{}", torch::cuda::is_available());
    auto tensor1 = torch::randn({ 2, 3, 4, 5 }).to(torch::kCUDA);
    auto tensor2 = torch::randn({ 2, 3, 4, 5 }).to(torch::kCUDA);
    lifuren::logTensor("tensor1", tensor1);
    lifuren::logTensor("tensor2", tensor2);
    lifuren::logTensor("tensor1 + tensor2", tensor1 + tensor2);
}

[[maybe_unused]] static void testLayer() {
    const size_t size = 24;
    float data[size] { 0.0F };
    std::for_each(data, data + size, [i = 0.0F](auto& v) mutable {
        // v = ++i;
        i += 0.001F;
        v = i;
    });
    // N C H W
    torch::Tensor a = torch::from_blob(data, {2, 2, 2, 3}, torch::kFloat32).clone();
    lifuren::logTensor("a", a);
    // N C L
    torch::Tensor b = torch::from_blob(data, {4, 2, 3}, torch::kFloat32).clone();
    lifuren::logTensor("b", b);
    torch::nn::LayerNorm ln(torch::nn::LayerNormOptions({ 2, 2, 3 }));
    lifuren::logTensor("ln", ln->forward(a));
    torch::nn::BatchNorm2d bn2d(torch::nn::BatchNorm2dOptions(2));
    lifuren::logTensor("bn2d", bn2d->forward(a));
    torch::nn::BatchNorm1d bn1d(torch::nn::BatchNorm1dOptions(2));
    lifuren::logTensor("bn1d", bn1d->forward(b));
}

[[maybe_unused]] static void testTensor() {
    // auto tensor = torch::randn({ 2, 5, 27 });
    // lifuren::logTensor("tensor", tensor);
    // // lifuren::logTensor("tensor", tensor.sizes());
    // auto tensors = tensor.split({24, 3}, 2);
    // lifuren::logTensor("tensor", tensors[0]);
    // lifuren::logTensor("tensor", tensors[1]);
    // // lifuren::logTensor("tensor", tensors[0].sizes());
    // // lifuren::logTensor("tensor", tensors[1].sizes());
    // auto tensor = torch::randn({ 2, 5, 4 });
    // auto a = tensor.slice(1, 0, 4);
    // auto z = tensor.slice(1, 1, 5);
    // // auto a = tensor.index_select(1, torch::tensor({0, 1, 2, 3}));
    // // auto z = tensor.index_select(1, torch::tensor({1, 2, 3, 4}));
    // lifuren::logTensor("tensor", tensor);
    // lifuren::logTensor("tensor", a);
    // lifuren::logTensor("tensor", z);
    auto tensor = torch::tensor({1, 0, 1, 0, 0, 0, 1, 1}, torch::kFloat32).reshape({2, 4});
    auto norm   = torch::nn::BatchNorm1d(2);
    lifuren::logTensor("tensor", norm(tensor));
}

LFR_TEST(
    // testJit();
    // testCuda();
    // testLayer();
    testTensor();
);

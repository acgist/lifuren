#include "lifuren/Test.hpp"

#include <fstream>

#include "torch/torch.h"
#include "torch/script.h"

#include "spdlog/spdlog.h"

#include "lifuren/File.hpp"
#include "lifuren/Layer.hpp"
#include "lifuren/Torch.hpp"
#include "lifuren/Config.hpp"

[[maybe_unused]] static void testTensor() {
    std::ofstream out(lifuren::file::join({lifuren::config::CONFIG.tmp, "tensor.data"}).string());
    for(int i = 0; i < 10; ++i) {
        auto tensor = torch::linspace(1, 40, 40).reshape({4, 2, 5});
        lifuren::write_tensor(out, tensor);
    }
    out.close();
    std::ifstream in(lifuren::file::join({lifuren::config::CONFIG.tmp, "tensor.data"}).string());
    int i = 0;
    while(true) {
        auto tensor = lifuren::read_tensor(in);
        if(in.eof()) {
            break;
        }
        lifuren::logTensor("tensor", tensor);
        SPDLOG_DEBUG("index : {}", i++);
    }
    in.close();
}

[[maybe_unused]] static void testLayer() {
    // （Batch    Normalization）批量归一化
    // （Layer    Normalization）层归一化
    // （Instance Normalization）实例归一化
    // （Group    Normalization）组归一化
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

[[maybe_unused]] static void testLoss() {
    torch::nn::MSELoss loss;
    // torch::nn::CrossEntropyLoss loss;
    // auto a = torch::randn({ 100 });
    // auto b = torch::randn({ 100 });
    auto a = torch::randn({ 7, 100 });
    auto b = torch::randn({ 7, 100 });
    a.requires_grad_(true);
    b.requires_grad_(true);
    auto c = loss(a, b);
    c.backward();
    lifuren::logTensor("a", a.sizes());
    lifuren::logTensor("b", b.sizes());
    lifuren::logTensor("c", c.sizes());
}

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

LFR_TEST(
    testTensor();
    // testLayer();
    // testLoss();
    // testJit();
);

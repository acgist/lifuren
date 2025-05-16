#include "lifuren/Test.hpp"

#include "torch/torch.h"
#include "torch/script.h"

#include "spdlog/spdlog.h"

#include "lifuren/File.hpp"
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
    torch::nn::BatchNorm1d bn1d(torch::nn::BatchNorm1dOptions(2));
    lifuren::logTensor("bn1d", bn1d->forward(b));
    torch::nn::BatchNorm2d bn2d(torch::nn::BatchNorm2dOptions(2));
    lifuren::logTensor("bn2d", bn2d->forward(a));
}

[[maybe_unused]] static void testTensor() {
    // auto tensor = torch::randn({ 2, 3 });
    // // auto tensor = torch::randn({ 2, 3, 3 });
    // lifuren::logTensor("tensor", tensor);
    // lifuren::logTensor("tensor", tensor.index({ 0 }));
    // lifuren::logTensor("tensor", tensor.index({ 1 }));
    // lifuren::logTensor("tensor", tensor.index({ "...", 0 }));
    // lifuren::logTensor("tensor", tensor.index({ "...", 1 }));
    // lifuren::logTensor("tensor", tensor.select(0, 0));
    // lifuren::logTensor("tensor", tensor.select(0, 1));
    // auto xxxxxx = torch::randn({ 2, 3 });
    // lifuren::logTensor("xxxxxx", xxxxxx);
    // lifuren::logTensor("cat tensor", torch::cat({ tensor, xxxxxx },  0));
    // lifuren::logTensor("cat tensor", torch::cat({ tensor, xxxxxx },  1));
    // lifuren::logTensor("cat tensor", torch::cat({ tensor, xxxxxx }, -1));
    // lifuren::logTensor("stack tensor", torch::stack({ tensor, xxxxxx },  0));
    // lifuren::logTensor("stack tensor", torch::stack({ tensor, xxxxxx },  1));
    // lifuren::logTensor("stack tensor", torch::stack({ tensor, xxxxxx }, -1));
    // lifuren::logTensor("stack tensor", torch::stack({ tensor, xxxxxx }, -1).index({ "...", 0 }));
    // lifuren::logTensor("stack tensor", torch::stack({ tensor, xxxxxx }, -1).index({ "...", 1 }));
    // -
    // float l[] = { 1, 2, 3, 4 };
    // // 错误
    // // auto tensor = torch::from_blob(l, { 4 }, torch::kFloat16).clone();
    // // 正确
    // auto tensor = torch::from_blob(l, { 4 }, torch::kFloat32).to(torch::kFloat16).clone();
    // lifuren::logTensor("tensor", tensor);
    // -
    // torch::Tensor tensor = torch::range(1, 36, torch::kFloat32).reshape({2, 3, 2, 3});
    // lifuren::logTensor("tensor", tensor);
    // auto a = tensor.slice(1, 0, 1);
    // lifuren::logTensor("tensor", a.squeeze());
    // lifuren::logTensor("tensor", a.squeeze().unsqueeze(1));
    // lifuren::logTensor("tensor", a.mul(tensor));
    // -
    // [100, 3, 16, 8]
    // [100,    16, 8]
    torch::Tensor a = torch::ones({2, 3, 2, 3});
    torch::Tensor b = torch::ones({2,    2, 3});
    torch::Tensor c = torch::ones({2,    2, 3}) * 2;
    torch::Tensor d = torch::ones({2,    2, 3}) * 3;
    lifuren::logTensor("a", a);
    // auto b1 = a.select(1, 0).mul(b);
    // auto c1 = a.select(1, 1).mul(c);
    // auto d1 = a.select(1, 2).mul(d);
    int i = 0;
    auto array = a.split(1, 1);
    std::vector<torch::Tensor> vector;
    for(auto iter = array.begin(); iter != array.end(); ++i, ++iter) {
        if(i % 3 == 0) {
            lifuren::logTensor("a", iter->squeeze(1).sizes());
            lifuren::logTensor("a", torch::transpose(b, 1, 2).sizes());
            vector.push_back(iter->squeeze(1).matmul(torch::transpose(b, 1, 2)));
        } else if(i % 3 == 1) {
            vector.push_back(iter->squeeze(1).matmul(torch::transpose(c, 1, 2)));
        } else {
            vector.push_back(iter->squeeze(1).matmul(torch::transpose(d, 1, 2)));
        }
        lifuren::logTensor("a", *iter);
    }
    lifuren::logTensor("a", torch::stack(vector, 1));
    lifuren::logTensor("a", torch::stack(vector, 1).matmul(a));
}

LFR_TEST(
    // testJit();
    // testLayer();
    testTensor();
);

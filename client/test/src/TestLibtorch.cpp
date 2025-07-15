#include "lifuren/Test.hpp"

#include "torch/torch.h"
#include "torch/script.h"

#include "lifuren/File.hpp"
#include "lifuren/Layer.hpp"
#include "lifuren/Config.hpp"

[[maybe_unused]] static void testJit() {
    auto model = torch::jit::load(lifuren::file::join({ lifuren::config::CONFIG.tmp, "lifuren.pth" }).string());
    std::vector<torch::jit::IValue> inputs;
    auto input = torch::randn({ 1 });
    inputs.push_back(std::move(input));
    model.eval();
    auto tensor = model.forward(inputs);
    auto result = tensor.toTensor().template item<float>();
    std::cout << result << std::endl;
}

[[maybe_unused]] static void testLayer() {
    lifuren::nn::AttentionBlock attention(32, 4, 200);
    auto input  = torch::randn({100, 32, 10, 20});
    auto output = attention->forward(input);
    std::cout << input.sizes() << std::endl;
    std::cout << output.sizes() << std::endl;
    torch::nn::MultiheadAttention attn(200, 10);
    output = torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 32 * 3, 1))->forward(input);
    std::cout << output.sizes() << std::endl;
    output = output.reshape({ 100, -1, 10 * 20 });
    std::cout << output.sizes() << std::endl;
    auto qkv = output.permute({1, 0, 2}).chunk(3, 0);
    auto q   = qkv[0];
    auto k   = qkv[1];
    auto v   = qkv[2];
    std::cout << q.sizes() << std::endl;
    std::cout << k.sizes() << std::endl;
    std::cout << v.sizes() << std::endl;
    auto [ o1, o2 ] = attn->forward(q, k, v);
    std::cout << o1.sizes() << std::endl;
    std::cout << o1.permute({1, 0, 2}).reshape({ 100, -1, 10, 20 }).sizes() << std::endl;
    std::cout << o2.sizes() << std::endl;
}

[[maybe_unused]] static void testTensor() {
    auto tensor = torch::arange(0, 18).reshape({ 2, 3, 3 });
    std::cout << "tensor\n" << tensor << std::endl;
    std::cout << "tensor\n" << tensor.index({ 0 }) << std::endl;
    std::cout << "tensor\n" << tensor.index({ 1 }) << std::endl;
    std::cout << "tensor\n" << tensor.index({ "...", 0 }) << std::endl;
    std::cout << "tensor\n" << tensor.index({ "...", 1 }) << std::endl;
    std::cout << "tensor\n" << tensor.select(0, 0) << std::endl;
    std::cout << "tensor\n" << tensor.select(0, 1) << std::endl;
}

LFR_TEST(
    // testJit();
    // testLayer();
    testTensor();
);

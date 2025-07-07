#include "lifuren/Test.hpp"

#include "torch/torch.h"
#include "torch/script.h"

#include "spdlog/spdlog.h"

#include "opencv2/opencv.hpp"

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
    lifuren::log_tensor("result", result);
}

[[maybe_unused]] static void testLayer() {
    // auto input = torch::randn({100, 32, 10, 20});
    // lifuren::nn::Downsample layer(32, 32, false);
    // auto output = layer->forward(input);
    // lifuren::nn::Upsample layer(32, 32, false);
    // auto output = layer->forward(input);
    // lifuren::log_tensor("size", output.sizes());
    // lifuren::nn::TimeEmbedding layer(10, 4, 100);
    // auto input = torch::arange(0, 10);
    // // auto input = torch::randint(10, { 4 });
    // auto output = layer->forward(input);
    // lifuren::log_tensor("input", input);
    // lifuren::log_tensor("output", output);
    lifuren::nn::AttentionBlock attention(32, 4, 200);
    auto input  = torch::randn({100, 32, 10, 20});
    auto output = attention->forward(input);
    std::cout << input.sizes() << std::endl;
    std::cout << output.sizes() << std::endl;
    torch::nn::MultiheadAttention attn(200, 10);
    std::cout << "====" << std::endl;
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
    // auto tensor = torch::randn({ 2, 3 });
    // // auto tensor = torch::randn({ 2, 3, 3 });
    // lifuren::log_tensor("tensor", tensor);
    // lifuren::log_tensor("tensor", tensor.index({ 0 }));
    // lifuren::log_tensor("tensor", tensor.index({ 1 }));
    // lifuren::log_tensor("tensor", tensor.index({ "...", 0 }));
    // lifuren::log_tensor("tensor", tensor.index({ "...", 1 }));
    // lifuren::log_tensor("tensor", tensor.select(0, 0));
    // lifuren::log_tensor("tensor", tensor.select(0, 1));
    // auto xxxxxx = torch::randn({ 2, 3 });
    // lifuren::log_tensor("xxxxxx", xxxxxx);
    // lifuren::log_tensor("cat tensor", torch::cat({ tensor, xxxxxx },  0));
    // lifuren::log_tensor("cat tensor", torch::cat({ tensor, xxxxxx },  1));
    // lifuren::log_tensor("cat tensor", torch::cat({ tensor, xxxxxx }, -1));
    // lifuren::log_tensor("stack tensor", torch::stack({ tensor, xxxxxx },  0));
    // lifuren::log_tensor("stack tensor", torch::stack({ tensor, xxxxxx },  1));
    // lifuren::log_tensor("stack tensor", torch::stack({ tensor, xxxxxx }, -1));
    // lifuren::log_tensor("stack tensor", torch::stack({ tensor, xxxxxx }, -1).index({ "...", 0 }));
    // lifuren::log_tensor("stack tensor", torch::stack({ tensor, xxxxxx }, -1).index({ "...", 1 }));
    // -
    // float l[] = { 1, 2, 3, 4 };
    // // 错误
    // // auto tensor = torch::from_blob(l, { 4 }, torch::kFloat16).clone();
    // // 正确
    // auto tensor = torch::from_blob(l, { 4 }, torch::kFloat32).to(torch::kFloat16).clone();
    // lifuren::log_tensor("tensor", tensor);
    // -
    // torch::Tensor tensor = torch::range(1, 36, torch::kFloat32).reshape({2, 3, 2, 3});
    // lifuren::log_tensor("tensor", tensor);
    // auto a = tensor.slice(1, 0, 1);
    // lifuren::log_tensor("tensor", a.squeeze());
    // lifuren::log_tensor("tensor", a.squeeze().unsqueeze(1));
    // lifuren::log_tensor("tensor", a.mul(tensor));
    // -
    // torch::Tensor a = torch::ones({2, 6, 2, 2});
    // torch::Tensor b = torch::ones({2, 2, 2, 2});
    // // torch::Tensor b = torch::ones({2, 1, 2, 3});
    // // torch::Tensor b = torch::ones({2, 3, 2, 3});
    // lifuren::log_tensor("a", a);
    // lifuren::log_tensor("b", b);
    // // lifuren::log_tensor("c", a.mul(b));
    // lifuren::log_tensor("c", a.matmul(b));
    // -
    // torch::Tensor a = torch::arange(0, 2 * 16 * 9).reshape({2, 1, 16, 9});
    // lifuren::log_tensor("a", a);
    // lifuren::log_tensor("a", torch::transpose(a, 2, 3));
    // lifuren::log_tensor("a", a                        .reshape({ 2, 4, 4,  9 }).permute({ 0, 1, 3, 2 }).reshape({ 2, 12, 3, 4 }).permute({ 0, 1, 3, 2 }));
    // lifuren::log_tensor("a", torch::transpose(a, 2, 3).reshape({ 2, 3, 3, 16 }).permute({ 0, 1, 3, 2 }).reshape({ 2, 12, 4, 3 }).permute({ 0, 1, 3, 2 }).transpose(2, 3));
    // lifuren::log_tensor("a", a == a
    //     .reshape({ 2, 4, 4, 9 }).permute({ 0, 1, 3, 2 }).reshape({ 2, 12, 3, 4 }).permute({ 0, 1,  3, 2 })
    //     .permute({ 0, 1, 3, 2 }).reshape({ 2, 4, 9, 4 }).permute({ 0,  1, 3, 2 }).reshape({ 2, 1, 16, 9 })
    // );
    // lifuren::log_tensor("a", a == a
    //     .transpose(2, 3).reshape({ 2, 3, 3, 16 }).permute({ 0, 1,  3, 2 }).reshape({ 2, 12, 4, 3 }).permute({ 0, 1, 3,  2 }).transpose(2, 3)
    //     .transpose(2, 3).permute({ 0, 1, 3,  2 }).reshape({ 2, 3, 16, 3 }).permute({ 0,  1, 3, 2 }).reshape({ 2, 1, 9, 16 }).transpose(2, 3)
    // );
    // -
    // torch::Tensor a = torch::arange(0, 16 * 2 * 3).reshape({16, 2, 3});
    // std::cout << a.data_ptr() << std::endl;
    // for(int i = 0; i < 16 - 5; ++i) {
    //     auto b = a.slice(0, i,     i + 5    );
    //     auto c = a.slice(0, i + 5, i + 5 + 1);
    //     std::cout << b.data_ptr() << " = " << b << std::endl;
    //     std::cout << c.data_ptr() << " = " << c << std::endl;
    // }
    // -
    // torch::Tensor a = torch::arange(0, 64).reshape({ 4, 4, 4 });
    // std::cout << a << std::endl;
    // std::cout << a.index_select(0, torch::tensor({0, 1})) << std::endl;
    // std::cout << a.index_select(0, torch::tensor({0, 2})) << std::endl;
    // std::cout << a.index_select(0, torch::tensor({1, 2})) << std::endl;
    // -
    auto tensor = torch::arange(0, 10 * 8 * 4).reshape({ 10, 8, 4 });
    std::cout << tensor.sizes() << std::endl;
    tensor = tensor.select(1, 0).select(1, 0);
    std::cout << tensor << std::endl;
}

LFR_TEST(
    // testJit();
    // testLayer();
    testTensor();
);

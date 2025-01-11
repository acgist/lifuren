#include "lifuren/Test.hpp"

#include <functional>

#include "torch/torch.h"

#include "spdlog/fmt/ostr.h"

LFR_FORMAT_LOG_STREAM(at::Tensor);
LFR_FORMAT_LOG_STREAM(c10::IntArrayRef);

[[maybe_unused]] static void testPrint() {
    torch::Tensor tensor = torch::randn({ 2, 4 });
    SPDLOG_DEBUG("\n{}", tensor);
}

[[maybe_unused]] static void testTensor() {
    const size_t size = 24;
    float data[size] { 0.0F };
    std::for_each(data, data + size, [i = 0.0F](auto& v) mutable {
        v = ++i;
    });
    torch::Tensor a = torch::from_blob(data, {4, 6}, torch::kFloat32);
    // torch::Tensor a = torch::rand({4, 6});
    SPDLOG_DEBUG("\n{}", a);
    SPDLOG_DEBUG("\n{}", a.t());
    SPDLOG_DEBUG("\n{}", a.numel());
    SPDLOG_DEBUG("\n{}", a.element_size());
    SPDLOG_DEBUG("\n{}", a.flatten());
    SPDLOG_DEBUG("\n{}", a.reshape({6, 4}));
    SPDLOG_DEBUG("\n{}", a.permute({1, 0}));
    SPDLOG_DEBUG("\n{}", torch::tensor({1.0F, 2.0F, 3.0F}, torch::kFloat32));
    SPDLOG_DEBUG("zero: {}", torch::zeros({10}));
    SPDLOG_DEBUG("zero: {}", a.sizes()[0]);
    SPDLOG_DEBUG("zero: {}", a.sizes()[1]);
}

[[maybe_unused]] static void testNorm() {
    // 批量归一化（Batch    Normalization）
    //   层归一化（Layer    Normalization）
    // 实例归一化（Instance Normalization）
    //   组归一化（Group    Normalization）
    const size_t size = 24;
    float data[size] { 0.0F };
    std::for_each(data, data + size, [i = 0.0F](auto& v) mutable {
        v = ++i;
    });
    // N C H W
    torch::Tensor a = torch::from_blob(data, {2, 2, 2, 3}, torch::kFloat32);
    SPDLOG_DEBUG("\n{}", a);
    // C
    torch::nn::LayerNorm   ln(torch::nn::LayerNormOptions({ 2, 3 }));
    SPDLOG_DEBUG("ln:\n{}", ln->forward(a));
    // N
    torch::nn::BatchNorm2d bn(torch::nn::BatchNorm2dOptions(2));
    SPDLOG_DEBUG("bn:\n{}", bn->forward(a));
}

[[maybe_unused]] static void testCat() {
    float data[] { 1.0F, 2.0F, 3.0F, 4.0F };
    torch::Tensor a = torch::from_blob(data, { 2, 2 }, torch::kFloat32).clone();
    torch::Tensor b = torch::from_blob(data, { 2, 2 }, torch::kFloat32).clone();
    torch::Tensor c = torch::from_blob(data, { 2, 2 }, torch::kFloat32).clone();
    torch::Tensor d = torch::from_blob(data, { 2, 2 }, torch::kFloat32).clone();
    auto e = torch::cat({ a, b, c, d });
    auto f = torch::stack({a, b, c, d});
    SPDLOG_DEBUG("e size: {}", e.sizes());
    SPDLOG_DEBUG("f size: {}", f.sizes());
}

[[maybe_unused]] static void testLinear() {
    auto input  = torch::randn({4, 2, 8});
    auto linear1 = torch::nn::Linear(torch::nn::LinearOptions(16, 36).bias(true));
    auto linear2 = torch::nn::Linear(torch::nn::LinearOptions(16, 36).bias(true));
    auto linear3 = torch::nn::Linear(torch::nn::LinearOptions(16, 36).bias(true));
    auto linear4 = torch::nn::Linear(torch::nn::LinearOptions(16, 36).bias(true));
    // SPDLOG_DEBUG("input ：{}", input);
    // SPDLOG_DEBUG("input ：{}", input.select(0, 0).view({16}));
    // SPDLOG_DEBUG("input ：{}", input.select(0, 0).flatten());
    // SPDLOG_DEBUG("input ：{}", input.select(0, 1));
    // SPDLOG_DEBUG("input ：{}", input.select(0, 2));
    // SPDLOG_DEBUG("input ：{}", input.select(0, 3));
    auto output = torch::stack({
        linear1->forward(input.select(0, 0).flatten()).add(linear2->forward(input.select(0, 1).flatten())).view({2, 18}),
        linear3->forward(input.select(0, 2).flatten()).mul(linear4->forward(input.select(0, 3).flatten())).view({2, 18})
    });
    SPDLOG_DEBUG("input ：{}", input);
    SPDLOG_DEBUG("output：{}", output);
    SPDLOG_DEBUG("input ：{}", input.sizes());
    SPDLOG_DEBUG("output：{}", output.sizes());
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
    SPDLOG_DEBUG("{}", a.sizes());
    SPDLOG_DEBUG("{}", b.sizes());
    SPDLOG_DEBUG("{}", c.sizes());
}

[[maybe_unused]] static void testConv2d() {
    auto input  = torch::randn({2, 2, 18});
    auto conv2d = torch::nn::Conv2d(torch::nn::Conv2dOptions(2, 4, {1, 3}).stride({1, 2}).bias(true));
    auto output = conv2d->forward(input);
    SPDLOG_DEBUG("input ：{}", input);
    SPDLOG_DEBUG("oouput：{}", output);
    SPDLOG_DEBUG("input ：{}", input.sizes());
    SPDLOG_DEBUG("oouput：{}", output.sizes());
}

[[maybe_unused]] static void testSlice() {
    auto input  = torch::randn({2, 2, 2});
    SPDLOG_DEBUG("input ：{}", input);
    SPDLOG_DEBUG("input ：{}", input.slice(1, 0, 1));
    SPDLOG_DEBUG("input ：{}", input.slice(1, 1, 2));
}

[[maybe_unused]] static void testPermute() {
    auto input  = torch::ones({2, 2, 2}).to(torch::kFloat);
    auto linear = torch::nn::Linear(torch::nn::LinearOptions(2, 2));
    SPDLOG_DEBUG("output ：{}", input);
    SPDLOG_DEBUG("output ：{}", input.permute({0, 2, 1}));
    auto output = linear->forward(input.permute({0, 2, 1}));
    SPDLOG_DEBUG("output ：{}", output);
    SPDLOG_DEBUG("output ：{}", output.permute({0, 2, 1}));
}

[[maybe_unused]] static void testGRU() {
    // https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
    torch::nn::GRUOptions options(10, 20); // input_size hidden_size
    // options.num_layers(2);
    torch::nn::GRU gru(options);
    // input // L N input_size = 句子长度 批量数量 词语维度
    // auto i0 = torch::randn({ 5, 3, 10 }); // L N input_size
    // auto h0 = torch::randn({ 1, 3, 20 }); // D[1|2] * num_layers N hidden_size
    // auto [o1, h1] = gru->forward(i0, h0);
    // SPDLOG_DEBUG("o sizes:\n{}", o1.sizes());
    // SPDLOG_DEBUG("h sizes:\n{}", h1.sizes());
    auto i0 = torch::randn({ 5, 1, 10 });
    auto h0 = torch::randn({ 1, 1, 20 });
    auto [o1, h1] = gru->forward(i0, h0);
    SPDLOG_DEBUG("o1 sizes:\n{}", o1.sizes());
    SPDLOG_DEBUG("h1 sizes:\n{}", h1.sizes());
    auto i1 = torch::randn({ 5, 1, 10 });
    auto [o2, h2] = gru->forward(i1, h1);
    SPDLOG_DEBUG("o2 sizes:\n{}", o2.sizes());
    SPDLOG_DEBUG("h2 sizes:\n{}", h2.sizes());
}

[[maybe_unused]] static void testLSTM() {
    // https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
    torch::nn::LSTMOptions options(2, 4);
    torch::nn::LSTM lstm(options);
    auto input = torch::randn({ 5, 3, 2 });
    auto h0    = torch::randn({ 1, 3, 4 });
    auto c0    = torch::randn({ 1, 3, 4 });
    auto [output, v] = lstm->forward(input, std::make_tuple<>(h0, c0));
    auto [hn, cn] = v;
    // auto [output, [hn, cn]] = lstm->forward(input, std::make_tuple<>(h0, c0));
    SPDLOG_DEBUG("o sizes:\n{}", output.sizes());
    SPDLOG_DEBUG("h sizes:\n{}", hn.sizes());
    SPDLOG_DEBUG("c sizes:\n{}", cn.sizes());
}

LFR_TEST(
    // testPrint();
    // testTensor();
    // testNorm();
    // testCat();
    // testLinear();
    // testLoss();
    // testConv2d();
    // testSlice();
    testPermute();
    // testGRU();
    // testLSTM();
);

#include "lifuren/Test.hpp"

#include "torch/torch.h"
#include "torch/script.h"

#include "spdlog/spdlog.h"

#include "lifuren/File.hpp"
#include "lifuren/Torch.hpp"

[[maybe_unused]] static void testPrint() {
    torch::Tensor tensor = torch::randn({ 2, 4 });
    lifuren::logTensor("tensor", tensor);
}

[[maybe_unused]] static void testTensor() {
    // 函数
    // const size_t size = 24;
    // float data[size] { 0.0F };
    // std::for_each(data, data + size, [i = 0.0F](auto& v) mutable {
    //     v = ++i;
    // });
    // torch::Tensor a = torch::from_blob(data, {4, 6}, torch::kFloat32);
    // lifuren::logTensor("a", a);
    // lifuren::logTensor("a.t", a.t());
    // lifuren::logTensor("a.numel", a.numel());
    // lifuren::logTensor("a.element_size", a.element_size());
    // lifuren::logTensor("a.flatten", a.flatten());
    // lifuren::logTensor("a.reshape", a.reshape({6, 4}));
    // lifuren::logTensor("a.permute", a.permute({1, 0}));
    // lifuren::logTensor("tensor", torch::tensor({1.0F, 2.0F, 3.0F}, torch::kFloat32));
    // lifuren::logTensor("zero", torch::zeros({10}));
    // lifuren::logTensor("zero", a.sizes()[0]);
    // lifuren::logTensor("zero", a.sizes()[1]);
    // 计算
    // torch::Tensor a = torch::rand({2, 4, 6});
    // torch::Tensor b = torch::ones({6});
    // b[1] = 0;
    // torch::Tensor c = torch::ones({4});
    // c[1] = 0;
    // torch::Tensor d = torch::ones({4, 6});
    // d = d.t().mul(c).t();
    // lifuren::logTensor("b", b);
    // lifuren::logTensor("c", c);
    // lifuren::logTensor("d", d);
    // lifuren::logTensor("a", a);
    // lifuren::logTensor("a * b", a.mul(b));
    // lifuren::logTensor("a * d", a.mul(d));
    // lifuren::logTensor("a sum", a.sum());
    // lifuren::logTensor("a sum 0", a.sum(0));
    // lifuren::logTensor("a sum 1", a.sum(1));
    // lifuren::logTensor("a sum 2", a.sum(2));
    torch::Tensor a = torch::rand({4, 6});
    torch::Tensor b = torch::rand({4, 6});
    torch::Tensor c = torch::rand({6, 4});
    lifuren::logTensor("a", a);
    lifuren::logTensor("b", b);
    lifuren::logTensor("c", c);
    lifuren::logTensor("a * b", a.mul(b));
    lifuren::logTensor("a * c", a.matmul(c));
}

[[maybe_unused]] static void testLayer() {
    // （Batch    Normalization）批量归一化
    // （Layer    Normalization）层归一化
    // （Instance Normalization）实例归一化
    // （Group    Normalization）组归一化
    const size_t size = 24;
    float data[size] { 0.0F };
    std::for_each(data, data + size, [i = 0.0F](auto& v) mutable {
        v = ++i;
    });
    // N C H W
    torch::Tensor a = torch::from_blob(data, {2, 2, 2, 3}, torch::kFloat32);
    lifuren::logTensor("a", a);
    // C
    torch::nn::LayerNorm   ln(torch::nn::LayerNormOptions({ 2, 3 }));
    lifuren::logTensor("ln", ln->forward(a));
    // N
    torch::nn::BatchNorm2d bn(torch::nn::BatchNorm2dOptions(2));
    lifuren::logTensor("bn", bn->forward(a));
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

[[maybe_unused]] static void testGRU() {
    // https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
    torch::nn::GRUOptions options(10, 20); // input_size hidden_size
    // options.num_layers(2);
    torch::nn::GRU gru(options);
    // input // L N input_size = 句子长度 批量数量 词语维度
    // auto i0 = torch::randn({ 5, 3, 10 }); // L N input_size
    // auto h0 = torch::randn({ 1, 3, 20 }); // D[1|2] * num_layers N hidden_size
    // auto [o1, h1] = gru->forward(i0, h0);
    // lifuren::logTensor("o sizes", o1.sizes());
    // lifuren::logTensor("h sizes", h1.sizes());
    auto i0 = torch::randn({ 5, 1, 10 });
    auto h0 = torch::randn({ 1, 1, 20 });
    auto [o1, h1] = gru->forward(i0, h0);
    lifuren::logTensor("o1 sizes", o1.sizes());
    lifuren::logTensor("h1 sizes", h1.sizes());
    auto i1 = torch::randn({ 5, 1, 10 });
    auto [o2, h2] = gru->forward(i1, h1);
    lifuren::logTensor("o2 sizes", o2.sizes());
    lifuren::logTensor("h2 sizes", h2.sizes());
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
    lifuren::logTensor("o sizes", output.sizes());
    lifuren::logTensor("h sizes", hn.sizes());
    lifuren::logTensor("c sizes", cn.sizes());
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
    // testPrint();
    testTensor();
    // testLayer();
    // testLoss();
    // testGRU();
    // testLSTM();
    // testJit();
);

#include "lifuren/Test.hpp"

#include "torch/torch.h"

#include "spdlog/fmt/ostr.h"

LFR_FORMAT_LOG_STREAM(at::Tensor);
LFR_FORMAT_LOG_STREAM(c10::IntArrayRef);

[[maybe_unused]] static void testGRU() {
    // https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
    torch::nn::GRUOptions options(10, 20); // input_size output_size
    // options.num_layers(2);
    torch::nn::GRU gru(options);
    auto input = torch::randn({5, 3, 10}); // L N input_size
    auto h0    = torch::randn({1, 3, 20}); // D[1|2] * num_layers N output_size
    auto [output, hn] = gru->forward(input, h0);
    SPDLOG_DEBUG("o sizes:\n{}", output.sizes());
    SPDLOG_DEBUG("h sizes:\n{}", hn.sizes());
}

[[maybe_unused]] static void testLSTM() {
    // https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
    torch::nn::LSTMOptions options(2, 4);
    torch::nn::LSTM lstm(options);
    auto input = torch::randn({ 5, 3, 10 });
    auto h0    = torch::randn({ 1, 3, 10 });
    auto c0    = torch::randn({ 1, 3, 10 });
    // auto [output, v] = lstm->forward(input, std::make_tuple<>(h0, c0));
    // auto [hn, cn] = v;
    auto [output, [hn, cn]] = lstm->forward(input, std::make_tuple<>(h0, c0));
    SPDLOG_DEBUG("o sizes:\n{}", output.sizes());
    SPDLOG_DEBUG("h sizes:\n{}", hn.sizes());
    SPDLOG_DEBUG("c sizes:\n{}", cn.sizes());
}

LFR_TEST(
    // testGRU();
    testLSTM();
);

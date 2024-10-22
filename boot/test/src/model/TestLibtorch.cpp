#include "lifuren/Test.hpp"

#include <functional>

#include "torch/torch.h"

#include "spdlog/fmt/ostr.h"

LFR_FORMAT_LOG_STREAM(at::Tensor);

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

LFR_TEST(
    // testPrint();
    testTensor();
    // testNorm();
);

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
    SPDLOG_DEBUG("\n{}", a.flatten());
    SPDLOG_DEBUG("\n{}", a.reshape({6, 4}));
    SPDLOG_DEBUG("\n{}", a.permute({1, 0}));
}

LFR_TEST(
    testPrint();
    testTensor();
);

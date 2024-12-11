#include "lifuren/Test.hpp"

#include "torch/torch.h"
#include "torch/script.h"

#include "lifuren/File.hpp"

#include "spdlog/spdlog.h"

[[maybe_unused]] static void testJit() {
    auto model = torch::jit::load(lifuren::file::join({ lifuren::config::CONFIG.tmp, "lifuren.pth" }).string());
    std::vector<torch::jit::IValue> inputs;
    auto input = torch::randn({ 1 });
    inputs.push_back(std::move(input));
    model.eval();
    auto tensor = model.forward(inputs);
    auto result = tensor.toTensor().template item<float>();
    SPDLOG_DEBUG("{}", result);
}

LFR_TEST(
    testJit();
);

#include "lifuren/Test.hpp"

#include "spdlog/spdlog.h"

#include "lifuren/File.hpp"
#include "lifuren/Onnx.hpp"

[[maybe_unused]] static void testOnnx() {
    auto onnx_runtime = lifuren::OnnxRuntime("lifuren");
    onnx_runtime.createSession(lifuren::file::join({ lifuren::config::CONFIG.tmp, "lifuren.onnx" }).string());
    std::vector<float> input;
    input.push_back(1.0F);
    auto result = onnx_runtime.runSession(input.data(), input.size(), { 1 });
    SPDLOG_DEBUG("{}", result[0]);
    SPDLOG_DEBUG("{}", result.size());
}

LFR_TEST(
    testOnnx();
);

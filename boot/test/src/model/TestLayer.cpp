#include "lifuren/Test.hpp"

#include "ggml.h"

#include "lifuren/Layer.hpp"
#include "lifuren/Tensor.hpp"

[[maybe_unused]] static void testLinear() {
    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false
    };
    ggml_context* ctx = ggml_init(params);
    lifuren::layer::Linear linear(1, 1, ctx, "fc1", true);
    // lifuren::layer::Linear linear(1, 1, ctx, "fc1", false);
    // lifuren::layer::Linear linear(5, 1, ctx, "fc1");
    // lifuren::layer::Linear linear(5, 2, ctx, "fc1");
    // lifuren::layer::Linear linear(5, 4, ctx, "fc1");
    std::map<std::string, ggml_tensor*> weights;
    linear.defineWeight(weights);
    ggml_tensor* weight = weights["fc1.weight"];
    ggml_tensor* bias   = weights["fc1.bias"];
    // ggml_set_param(ctx, weight);
    // ggml_set_param(ctx, bias);
    lifuren::tensor::fillRange(weight, 2);
    if(bias != nullptr) {
        lifuren::tensor::fill(bias, 1.0F);
    }
    ggml_tensor* input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 10);
    // ggml_tensor* input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5, 5);
    ggml_tensor* output = linear.forward(input);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, 1024, true);
    ggml_build_forward_expand(gf, output);
    lifuren::tensor::fill(input, 2.0F);
    ggml_graph_compute_with_ctx(ctx, gf, 4);
    float* odata = ggml_get_data_f32(output);
    SPDLOG_DEBUG("结果大小：{} - {} - {} - {}", output->ne[0], output->ne[1], output->ne[2], output->ne[3]);
    for(int i = 0; i < ggml_nelements(output); ++i) {
        SPDLOG_DEBUG("计算结果：{} - {}", i, odata[i]);
    }
}

[[maybe_unused]] static void testConv2d() {
    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false
    };
    ggml_context* ctx = ggml_init(params);
    lifuren::layer::Conv2d conv2d(3, 4, 3, ctx);
    std::map<std::string, ggml_tensor*> weights;
    conv2d.defineWeight(weights);
    ggml_tensor* kernel = weights["conv2d.kernel"];
    ggml_tensor* bias   = weights["conv2d.bias"];
    // ggml_set_param(ctx, kernel);
    // ggml_set_param(ctx, bias);
    lifuren::tensor::fillRange(kernel, 2);
    if(bias != nullptr) {
        lifuren::tensor::fill(bias, 1.0F);
    }
    ggml_tensor* input  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 9, 9, 3, 10);
    ggml_tensor* output = conv2d.forward(input);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, 1024, true);
    ggml_build_forward_expand(gf, output);
    lifuren::tensor::fill(input, 2.0F);
    ggml_graph_compute_with_ctx(ctx, gf, 4);
    float* odata = ggml_get_data_f32(output);
    SPDLOG_DEBUG("结果大小：{} - {} - {} - {}", output->ne[0], output->ne[1], output->ne[2], output->ne[3]);
    for(int i = 0; i < ggml_nelements(output); ++i) {
        SPDLOG_DEBUG("计算结果：{} - {}", i, odata[i]);
    }
}

LFR_TEST(
    // testLinear();
    testConv2d();
);

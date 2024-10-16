#include "lifuren/Test.hpp"

#include "ggml.h"

#include "lifuren/Layer.hpp"
#include "lifuren/Tensor.hpp"

[[maybe_unused]] static void testGRU() {
    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false
    };
    ggml_context* ctx = ggml_init(params);
    auto gru = lifuren::layer::gru(2, 4, 10, ctx, ctx);
    gru->defineWeight();
    gru->initWeight([](auto w) {
        lifuren::tensor::fill(w, 1.0F);
    });
    ggml_tensor* input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 10);
    ggml_tensor* output = gru->forward(input);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, 1024, true);
    ggml_build_forward_expand(gf, output);
    lifuren::tensor::fill(input, 2.0F);
    ggml_graph_compute_with_ctx(ctx, gf, 4);
    lifuren::tensor::print(output);
    ggml_free(ctx);
}

[[maybe_unused]] static void testLSTM() {
    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false
    };
    ggml_context* ctx = ggml_init(params);
    auto lstm = lifuren::layer::lstm(2, 4, 10, ctx, ctx);
    lstm->defineWeight();
    lstm->initWeight([](auto w) {
        lifuren::tensor::fill(w, 1.0F);
    });
    ggml_tensor* input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 10);
    ggml_tensor* output = lstm->forward(input);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, 1024, true);
    ggml_build_forward_expand(gf, output);
    lifuren::tensor::fill(input, 2.0F);
    ggml_graph_compute_with_ctx(ctx, gf, 4);
    lifuren::tensor::print(output);
    ggml_free(ctx);
}

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
    linear.defineWeight();
    ggml_tensor* weight = linear["fc1.weight"];
    ggml_tensor* bias   = linear["fc1.bias"];
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
    lifuren::tensor::print(output);
    ggml_free(ctx);
}

[[maybe_unused]] static void testConv2d() {
    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false
    };
    ggml_context* ctx = ggml_init(params);
    lifuren::layer::Conv2d conv2d(3, 4, 3, ctx);
    conv2d.defineWeight();
    ggml_tensor* kernel = conv2d["conv2d.kernel"];
    ggml_tensor* bias   = conv2d["conv2d.bias"];
    // ggml_set_param(ctx, kernel);
    // ggml_set_param(ctx, bias);
    lifuren::tensor::fillRange(kernel, 2);
    if(bias != nullptr) {
        lifuren::tensor::fill(bias, 1.0F);
    }
    // ggml_tensor* input  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 3, 3, 2);
    ggml_tensor* input  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 9, 9, 3, 10);
    ggml_tensor* output = conv2d.forward(input);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, 1024, true);
    ggml_build_forward_expand(gf, output);
    lifuren::tensor::fill(input, 2.0F);
    ggml_graph_compute_with_ctx(ctx, gf, 4);
    float* odata = ggml_get_data_f32(output);
    lifuren::tensor::print(input);
    lifuren::tensor::print(output);
    ggml_free(ctx);
}

[[maybe_unused]] static void testAvgPool2d() {
    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false
    };
    ggml_context* ctx    = ggml_init(params);
    ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5, 5);
    ggml_tensor * output = lifuren::function::avgPool2d(2, input, ctx);
    ggml_cgraph * gf     = ggml_new_graph_custom(ctx, 1024, true);
    ggml_build_forward_expand(gf, output);
    lifuren::tensor::fillRange(input, 0);
    ggml_graph_compute_with_ctx(ctx, gf, 4);
    lifuren::tensor::print(input);
    lifuren::tensor::print(output);
    ggml_free(ctx);
}

[[maybe_unused]] static void testMaxPool2d() {
    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false
    };
    ggml_context* ctx    = ggml_init(params);
    ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5, 5);
    ggml_tensor * output = lifuren::function::maxPool2d(2, input, ctx);
    ggml_cgraph * gf     = ggml_new_graph_custom(ctx, 1024, true);
    ggml_build_forward_expand(gf, output);
    lifuren::tensor::fillRange(input, 0);
    ggml_graph_compute_with_ctx(ctx, gf, 4);
    lifuren::tensor::print(input);
    lifuren::tensor::print(output);
    ggml_free(ctx);
}

[[maybe_unused]] static void testLoss() {
    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false
    };
    ggml_context* ctx    = ggml_init(params);
    ggml_tensor * source = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 5);
    ggml_tensor * target = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 5);
    // ggml_tensor * loss   = lifuren::loss::l1Loss(ctx, source, target);
    ggml_tensor * loss   = lifuren::loss::bceLoss(ctx, ggml_sigmoid(ctx, source), target);
    // ggml_tensor * loss   = lifuren::loss::mseLoss(ctx, source, target);
    // ggml_tensor * loss   = lifuren::loss::crossEntropyLoss(ctx, source, target);
    ggml_cgraph * gf     = ggml_new_graph_custom(ctx, 1024, true);
    ggml_build_forward_expand(gf, loss);
    // lifuren::tensor::fill(source, 1);
    // lifuren::tensor::fill(target, 1);
    // lifuren::tensor::fillRange(source, 1);
    // lifuren::tensor::fillRange(target, 1);
    float aa[] = { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F };
    // float bb[] = { 2.0F, 4.0F, 3.0F, 4.0F, 5.0F };
    float bb[] = { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F };
    lifuren::tensor::fill(source, aa);
    lifuren::tensor::fill(target, bb);
    ggml_graph_compute_with_ctx(ctx, gf, 4);
    lifuren::tensor::print(source);
    lifuren::tensor::print(target);
    lifuren::tensor::print(loss);
    ggml_free(ctx);
}

[[maybe_unused]] static void testFunction() {
    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false
    };
    ggml_context* ctx    = ggml_init(params);
    // ggml_tensor * input  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 24);
    // ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 6);
    // ggml_tensor * input  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 2, 3);
    ggml_tensor * input  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 2, 2, 2);
    // ggml_tensor * output = lifuren::function::flatten(ctx, input);
    ggml_tensor * output = ggml_cont(ctx, input);
    ggml_cgraph * gf     = ggml_new_graph_custom(ctx, 1024, true);
    ggml_build_forward_expand(gf, output);
    lifuren::tensor::fillRange(input, 0);
    ggml_graph_compute_with_ctx(ctx, gf, 4);
    lifuren::tensor::print(input);
    lifuren::tensor::print(output);
    ggml_free(ctx);
}

LFR_TEST(
    // testGRU();
    // testLSTM();
    // testLinear();
    // testConv2d();
    // testAvgPool2d();
    // testMaxPool2d();
    // testLoss();
    testFunction();
);

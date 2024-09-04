#include "lifuren/Layers.hpp"

#include "lifuren/Logger.hpp"
#include "lifuren/Tensors.hpp"

#include "ggml.h"

#include "spdlog/spdlog.h"

static void testLinear() {
    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024,
        .mem_buffer = NULL,
    };
    ggml_context* ctx = ggml_init(params);
    lifuren::layers::Linear linear(1, 1, ctx, "fc1", true);
    // lifuren::layers::Linear linear(1, 1, ctx, "fc1", false);
    // lifuren::layers::Linear linear(5, 1, ctx, "fc1");
    // lifuren::layers::Linear linear(5, 2, ctx, "fc1");
    // lifuren::layers::Linear linear(5, 4, ctx, "fc1");
    std::map<std::string, ggml_tensor*> weights;
    linear.defineWeight(weights);
    ggml_tensor* weight = weights["fc1.weight"];
    ggml_tensor* bias   = weights["fc1.bias"];
    // ggml_set_param(ctx, weight);
    // ggml_set_param(ctx, bias);
    lifuren::tensors::fillRange(weight, 2);
    if(bias != nullptr) {
        lifuren::tensors::fill(bias, 1.0F);
    }
    ggml_tensor* input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 10);
    // ggml_tensor* input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5, 5);
    ggml_tensor* output = linear.forward(input);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, 1024, true);
    ggml_build_forward_expand(gf, output);
    lifuren::tensors::fill(input, 2.0F);
    ggml_graph_compute_with_ctx(ctx, gf, 4);
    float* odata = ggml_get_data_f32(output);
    SPDLOG_DEBUG("结果大小：{} - {} - {} - {}", output->ne[0], output->ne[1], output->ne[2], output->ne[3]);
    for(int i = 0; i < ggml_nelements(output); ++i) {
        SPDLOG_DEBUG("计算结果：{} - {}", i, odata[i]);
    }
}

int main() {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    testLinear();
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}
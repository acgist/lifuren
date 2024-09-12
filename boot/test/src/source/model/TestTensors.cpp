#include "Test.hpp"

#include "lifuren/Tensors.hpp"

static void testMul() {
    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024,
        .mem_buffer = NULL,
    };
    ggml_context* ctx = ggml_init(params);
    // ggml_tensor * a   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor * a   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
    // ggml_tensor * b   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 10);
    ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 10, 4);
    ggml_tensor * c   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    // ggml_tensor * c   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 10);
    ggml_set_param(ctx, a);
    ggml_set_param(ctx, b);
    ggml_set_param(ctx, c);
    SPDLOG_DEBUG("a nbytes : {}", ggml_nbytes(a));
    lifuren::tensors::fill(a, 2.0F);
    lifuren::tensors::fill(b, 1.0F);
    lifuren::tensors::fill(c, 1.0F);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, 1024, true);
    ggml_tensor* r  = ggml_mul_mat(ctx, a, b);
    // ggml_tensor* r  = ggml_add(ctx, ggml_mul_mat(ctx, a, b), c);
    ggml_build_forward_expand(gf, r);
    ggml_graph_compute_with_ctx(ctx, gf, 4);
    float* rr = ggml_get_data_f32(r);
    SPDLOG_DEBUG("r ne : {} - {} - {} - {}", r->ne[0], r->ne[1], r->ne[2], r->ne[3]);
    for(size_t i = 0; i < ggml_nelements(r); ++i) {
        SPDLOG_DEBUG("r {} - {}", i, rr[i]);
    }
    ggml_free(ctx);
}

LFR_TEST(
    testMul();
);

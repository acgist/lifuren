#include "lifuren/Layer.hpp"

#include "fmt/format.h"

lifuren::layer::GRU::GRU(
    size_t input_size,
    size_t hidden_size,
    size_t batch_size,
    ggml_context* ctx,
    const std::string& name,
    const double dropout,
    const bool bias
) : Layer(ctx, ctx, name),
    input_size(input_size),
    hidden_size(hidden_size),
    batch_size(batch_size),
    dropout(dropout),
    bias_(bias)
{
}

lifuren::layer::GRU::GRU(
    size_t input_size,
    size_t hidden_size,
    size_t batch_size,
    ggml_context* ctx_weight,
    ggml_context* ctx_compute,
    const std::string& name,
    const double dropout,
    const bool bias
) : Layer(ctx_weight, ctx_compute, name),
    input_size(input_size),
    hidden_size(hidden_size),
    batch_size(batch_size),
    dropout(dropout),
    bias_(bias)
{
}

lifuren::layer::GRU::~GRU() {
}

std::string lifuren::layer::GRU::info() const {
    return fmt::format("{} => in = {} hd = {}", this->name, this->input_size, this->hidden_size);
}

ggml_tensor* lifuren::layer::GRU::forward(ggml_tensor* input) {
    // Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
    ggml_tensor* z = ggml_sigmoid(this->ctx_compute,
        ggml_add(this->ctx_compute,
            ggml_add(this->ctx_compute,
                ggml_mul_mat(this->ctx_compute, this->w_xz, input),
                ggml_mul_mat(this->ctx_compute, this->w_hz, this->h)
            ),
            this->b_z
        )
    );
    // R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
    ggml_tensor* r = ggml_sigmoid(this->ctx_compute,
        ggml_add(this->ctx_compute,
            ggml_add(this->ctx_compute,
                ggml_mul_mat(this->ctx_compute, this->w_xr, input),
                ggml_mul_mat(this->ctx_compute, this->w_hr, this->h)
            ),
            this->b_r
        )
    );
    // H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
    ggml_tensor* h_tilda = ggml_tanh(this->ctx_compute,
        ggml_add(this->ctx_compute,
            ggml_add(this->ctx_compute,
                ggml_mul_mat(this->ctx_compute, this->w_xh, input),
                ggml_mul_mat(this->ctx_compute,
                    this->w_hh,
                    ggml_mul(this->ctx_compute, r, this->h)
                )
            ),
            this->b_h
        )
    );
    // H = Z * H + (1 - Z) * H_tilda
    ggml_tensor* h = ggml_add(this->ctx_compute,
        ggml_mul(this->ctx_compute, z, this->h),
        ggml_mul(this->ctx_compute,
            ggml_add(this->ctx_compute,
                ggml_div(this->ctx_compute, z, z),
                ggml_neg(this->ctx_compute, z)
            ),
            h_tilda
        )
    );
    ggml_cpy(this->ctx_compute, h, this->h);
    // Y = H @ W_hq + b_q
    ggml_tensor* y = ggml_add(this->ctx_compute,
        ggml_mul_mat(this->ctx_compute, this->w_hq, this->h),
        this->b_q
    );
    return y;
}

void lifuren::layer::GRU::defineWeight() {
    // 更新门
    LFR_DEFINE_LAYER_2D(w_xz, input_size,  hidden_size)
    LFR_DEFINE_LAYER_2D(w_hz, hidden_size, hidden_size)
    // 重置门
    LFR_DEFINE_LAYER_2D(w_xr, input_size,  hidden_size)
    LFR_DEFINE_LAYER_2D(w_hr, hidden_size, hidden_size)
    // 候选隐状态
    LFR_DEFINE_LAYER_2D(w_xh, input_size,  hidden_size)
    LFR_DEFINE_LAYER_2D(w_hh, hidden_size, hidden_size)
    // 输出层
    LFR_DEFINE_LAYER_2D(w_hq, hidden_size, hidden_size)
    // 隐藏状态
    LFR_DEFINE_LAYER_2D(h, hidden_size, batch_size)
    if(this->bias_) {
        // 更新门
        LFR_DEFINE_LAYER_1D(b_z, hidden_size)
        // 重置门
        LFR_DEFINE_LAYER_1D(b_r, hidden_size)
        // 候选隐状态
        LFR_DEFINE_LAYER_1D(b_h, hidden_size)
        // 输出层
        LFR_DEFINE_LAYER_1D(b_q, hidden_size)
    }
}

void lifuren::layer::GRU::bindWeight(const std::map<std::string, ggml_tensor*>& weights) {
    LFR_BIND_WEIGHT(w_xz)
    LFR_BIND_WEIGHT(w_hz)
    LFR_BIND_WEIGHT(w_xr)
    LFR_BIND_WEIGHT(w_hr)
    LFR_BIND_WEIGHT(w_xh)
    LFR_BIND_WEIGHT(w_hh)
    LFR_BIND_WEIGHT(w_hq)
    LFR_BIND_WEIGHT(h)
    if(this->bias_) {
        LFR_BIND_WEIGHT(b_z)
        LFR_BIND_WEIGHT(b_r)
        LFR_BIND_WEIGHT(b_h)
        LFR_BIND_WEIGHT(b_q)
    }
}

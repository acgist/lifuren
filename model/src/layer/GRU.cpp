#include "lifuren/Layer.hpp"

#include "fmt/format.h"

lifuren::layer::GRU::GRU(
    size_t input_size,
    size_t hidden_size,
    ggml_context* ctx,
    const std::string& name,
    size_t num_layer,
    double dropout,
    bool   bias
) : Layer(ctx, ctx, name),
    input_size(input_size),
    hidden_size(hidden_size),
    num_layer(num_layer),
    dropout(dropout),
    bias_(bias)
{
}

lifuren::layer::GRU::GRU(
    size_t input_size,
    size_t hidden_size,
    ggml_context* ctx_weight,
    ggml_context* ctx_compute,
    const std::string& name,
    size_t num_layer,
    double dropout,
    bool   bias
) : Layer(ctx_weight, ctx_compute, name),
    input_size(input_size),
    hidden_size(hidden_size),
    num_layer(num_layer),
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
                ggml_mul_mat(this->ctx_compute, w_xz, input),
                ggml_mul_mat(this->ctx_compute, w_hz, this->h)
            ),
            b_z
        )
    );
    // R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
    ggml_tensor* r = ggml_sigmoid(this->ctx_compute,
        ggml_add(this->ctx_compute,
            ggml_add(this->ctx_compute,
                ggml_mul_mat(this->ctx_compute, w_xr, input),
                ggml_mul_mat(this->ctx_compute, w_hr, this->h)
            ),
            b_r
        )
    );
    // H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
    ggml_tensor* h_tilda = ggml_tanh(this->ctx_compute,
        ggml_add(this->ctx_compute,
            ggml_add(this->ctx_compute,
                ggml_mul_mat(this->ctx_compute, w_xh, input),
                ggml_mul_mat(this->ctx_compute,
                    this->w_hh,
                    ggml_mul(this->ctx_compute, r, this->h)
                )
            ),
            b_h
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
        ggml_mul_mat(this->ctx_compute, w_hq, this->h),
        this->b_q
    );
    return y;
}

void lifuren::layer::GRU::defineWeight(std::map<std::string, ggml_tensor*>& weights) {
    // 更新门
    this->w_xz = ggml_new_tensor_2d(this->ctx_weight, GGML_TYPE_F32, this->input_size,  this->hidden_size);
    this->w_hz = ggml_new_tensor_2d(this->ctx_weight, GGML_TYPE_F32, this->hidden_size, this->hidden_size);
    weights.emplace(this->name + ".w_xz", this->w_xz);
    weights.emplace(this->name + ".w_hz", this->w_hz);
    // 重置门
    this->w_xr = ggml_new_tensor_2d(this->ctx_weight, GGML_TYPE_F32, this->input_size,  this->hidden_size);
    this->w_hr = ggml_new_tensor_2d(this->ctx_weight, GGML_TYPE_F32, this->hidden_size, this->hidden_size);
    weights.emplace(this->name + ".w_xr", this->w_xr);
    weights.emplace(this->name + ".w_hr", this->w_hr);
    // 候选隐状态
    this->w_xh = ggml_new_tensor_2d(this->ctx_weight, GGML_TYPE_F32, this->input_size,  this->hidden_size);
    this->w_hh = ggml_new_tensor_2d(this->ctx_weight, GGML_TYPE_F32, this->hidden_size, this->hidden_size);
    weights.emplace(this->name + ".w_xh", this->w_xh);
    weights.emplace(this->name + ".w_hh", this->w_hh);
    // 隐藏状态
    this->h = ggml_new_tensor_2d(this->ctx_weight, GGML_TYPE_F32, this->hidden_size, this->batch_size);
    weights.emplace(this->name + ".h", this->h);
    // 输出
    this->w_hq = ggml_new_tensor_2d(this->ctx_weight, GGML_TYPE_F32, this->hidden_size, this->hidden_size);
    weights.emplace(this->name + ".w_hq", this->w_hq);
    if(this->bias_) {
        // 更新门
        this->b_z = ggml_new_tensor_1d(this->ctx_weight, GGML_TYPE_F32, this->hidden_size);
        weights.emplace(this->name + ".b_z", this->b_z);
        // 重置门
        this->b_r = ggml_new_tensor_1d(this->ctx_weight, GGML_TYPE_F32, this->hidden_size);
        weights.emplace(this->name + ".b_r", this->b_r);
        // 候选隐状态
        this->b_h = ggml_new_tensor_1d(this->ctx_weight, GGML_TYPE_F32, this->hidden_size);
        weights.emplace(this->name + ".b_h", this->b_h);
        // 输出
        this->b_q = ggml_new_tensor_1d(this->ctx_weight, GGML_TYPE_F32, this->hidden_size);
        weights.emplace(this->name + ".b_q", this->b_q);
    }
}

void lifuren::layer::GRU::bindWeight(const std::map<std::string, ggml_tensor*>& weights) {
    this->bindWeight(weights, this->name + ".w_xz", & this->w_xz);
    this->bindWeight(weights, this->name + ".w_hz", & this->w_hz);
    this->bindWeight(weights, this->name + ".w_xr", & this->w_xr);
    this->bindWeight(weights, this->name + ".w_hr", & this->w_hr);
    this->bindWeight(weights, this->name + ".w_xh", & this->w_xh);
    this->bindWeight(weights, this->name + ".w_hh", & this->w_hh);
    this->bindWeight(weights, this->name + ".h", & this->h);
    this->bindWeight(weights, this->name + ".w_hq", & this->w_hq);
    if(this->bias_) {
        this->bindWeight(weights, this->name + ".b_z", & this->b_z);
        this->bindWeight(weights, this->name + ".b_r", & this->b_r);
        this->bindWeight(weights, this->name + ".b_h", & this->b_h);
        this->bindWeight(weights, this->name + ".b_q", & this->b_q);
    }
}

#include "lifuren/Layer.hpp"

#include "fmt/format.h"

lifuren::layer::LSTM::LSTM(
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

lifuren::layer::LSTM::LSTM(
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

lifuren::layer::LSTM::~LSTM() {
}

std::string lifuren::layer::LSTM::info() const {
    return fmt::format("{} => in = {} hd = {}", this->name, this->input_size, this->hidden_size);
}

ggml_tensor* lifuren::layer::LSTM::forward(ggml_tensor* input) {
    // I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
    ggml_tensor* i = ggml_sigmoid(this->ctx_compute,
        ggml_add(this->ctx_compute,
            ggml_add(this->ctx_compute,
                ggml_mul_mat(this->ctx_compute, this->w_xi, input),
                ggml_mul_mat(this->ctx_compute, this->w_hi, this->h)
            ),
            this->b_i
        )
    );
    // F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
    ggml_tensor* f = ggml_sigmoid(this->ctx_compute,
        ggml_add(this->ctx_compute,
            ggml_add(this->ctx_compute,
                ggml_mul_mat(this->ctx_compute, this->w_xf, input),
                ggml_mul_mat(this->ctx_compute, this->w_hf, this->h)
            ),
            this->b_f
        )
    );
    // O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
    ggml_tensor* o = ggml_sigmoid(this->ctx_compute,
        ggml_add(this->ctx_compute,
            ggml_add(this->ctx_compute,
                ggml_mul_mat(this->ctx_compute, this->w_xo, input),
                ggml_mul_mat(this->ctx_compute, this->w_ho, this->h)
            ),
            this->b_o
        )
    );
    // C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
    ggml_tensor* c_tilda = ggml_tanh(this->ctx_compute,
        ggml_add(this->ctx_compute,
            ggml_add(this->ctx_compute,
                ggml_mul_mat(this->ctx_compute, this->w_xc, input),
                ggml_mul_mat(this->ctx_compute, this->w_hc, this->h)
            ),
            this->b_c
        )
    );
    // C = F * C + I * C_tilda
        ggml_mul(this->ctx_compute, f, this->c);
        ggml_mul(this->ctx_compute, i, c_tilda);
    ggml_tensor* c = ggml_add(this->ctx_compute,
        ggml_mul(this->ctx_compute, f, this->c),
        ggml_mul(this->ctx_compute, i, c_tilda)
    );
    ggml_cpy(this->ctx_compute, c, this->c);
    // H = O * torch.tanh(C)
    ggml_tensor* h = ggml_mul(this->ctx_compute,
        o,
        ggml_tanh(this->ctx_compute, c)
    );
    ggml_cpy(this->ctx_compute, h, this->h);
    // Y = (H @ W_hq) + b_q
    ggml_tensor* y = ggml_add(this->ctx_compute,
        ggml_mul_mat(this->ctx_compute, w_hq, this->h),
        this->b_q
    );
    return y;
}

void lifuren::layer::LSTM::defineWeight() {
    // 输入门
    this->w_xi = ggml_new_tensor_2d(this->ctx_weight, GGML_TYPE_F32, this->input_size,  this->hidden_size);
    this->w_hi = ggml_new_tensor_2d(this->ctx_weight, GGML_TYPE_F32, this->hidden_size, this->hidden_size);
    this->defineWeight(this->name + ".w_xi", this->w_xi);
    this->defineWeight(this->name + ".w_hi", this->w_hi);
    // 遗忘门
    this->w_xf = ggml_new_tensor_2d(this->ctx_weight, GGML_TYPE_F32, this->input_size,  this->hidden_size);
    this->w_hf = ggml_new_tensor_2d(this->ctx_weight, GGML_TYPE_F32, this->hidden_size, this->hidden_size);
    this->defineWeight(this->name + ".w_xf", this->w_xf);
    this->defineWeight(this->name + ".w_hf", this->w_hf);
    // 输出门
    this->w_xo = ggml_new_tensor_2d(this->ctx_weight, GGML_TYPE_F32, this->input_size,  this->hidden_size);
    this->w_ho = ggml_new_tensor_2d(this->ctx_weight, GGML_TYPE_F32, this->hidden_size, this->hidden_size);
    this->defineWeight(this->name + ".w_xo", this->w_xo);
    this->defineWeight(this->name + ".w_ho", this->w_ho);
    // 候选记忆元
    this->w_xc = ggml_new_tensor_2d(this->ctx_weight, GGML_TYPE_F32, this->input_size,  this->hidden_size);
    this->w_hc = ggml_new_tensor_2d(this->ctx_weight, GGML_TYPE_F32, this->hidden_size, this->hidden_size);
    this->defineWeight(this->name + ".w_xc", this->w_xc);
    this->defineWeight(this->name + ".w_hc", this->w_hc);
    // 输出层
    this->w_hq = ggml_new_tensor_2d(this->ctx_weight, GGML_TYPE_F32, this->hidden_size, this->hidden_size);
    this->defineWeight(this->name + ".w_hq", this->w_hq);
    // 隐藏状态
    this->h = ggml_new_tensor_2d(this->ctx_weight, GGML_TYPE_F32, this->hidden_size, this->batch_size);
    this->defineWeight(this->name + ".h", this->h);
    this->c = ggml_new_tensor_2d(this->ctx_weight, GGML_TYPE_F32, this->hidden_size, this->batch_size);
    this->defineWeight(this->name + ".c", this->c);
    if(this->bias_) {
        // 输入门
        this->b_i = ggml_new_tensor_1d(this->ctx_weight, GGML_TYPE_F32, this->hidden_size);
        this->defineWeight(this->name + ".b_i", this->b_i);
        // 遗忘门
        this->b_f = ggml_new_tensor_1d(this->ctx_weight, GGML_TYPE_F32, this->hidden_size);
        this->defineWeight(this->name + ".b_f", this->b_f);
        // 输出门
        this->b_o = ggml_new_tensor_1d(this->ctx_weight, GGML_TYPE_F32, this->hidden_size);
        this->defineWeight(this->name + ".b_o", this->b_o);
        // 候选记忆元
        this->b_c = ggml_new_tensor_1d(this->ctx_weight, GGML_TYPE_F32, this->hidden_size);
        this->defineWeight(this->name + ".b_c", this->b_c);
        // 输出层
        this->b_q = ggml_new_tensor_1d(this->ctx_weight, GGML_TYPE_F32, this->hidden_size);
        this->defineWeight(this->name + ".b_q", this->b_q);
    }
}


void lifuren::layer::LSTM::bindWeight(const std::map<std::string, ggml_tensor*>& weights) {
    this->bindWeight(weights, this->name + ".w_xi", & this->w_xi);
    this->bindWeight(weights, this->name + ".w_hi", & this->w_hi);
    this->bindWeight(weights, this->name + ".w_xf", & this->w_xf);
    this->bindWeight(weights, this->name + ".w_hf", & this->w_hf);
    this->bindWeight(weights, this->name + ".w_xo", & this->w_xo);
    this->bindWeight(weights, this->name + ".w_ho", & this->w_ho);
    this->bindWeight(weights, this->name + ".w_xc", & this->w_xc);
    this->bindWeight(weights, this->name + ".w_hc", & this->w_hc);
    this->bindWeight(weights, this->name + ".w_hq", & this->w_hq);
    this->bindWeight(weights, this->name + ".h", & this->h);
    this->bindWeight(weights, this->name + ".c", & this->c);
    if(this->bias_) {
        this->bindWeight(weights, this->name + ".b_i", & this->b_i);
        this->bindWeight(weights, this->name + ".b_f", & this->b_f);
        this->bindWeight(weights, this->name + ".b_o", & this->b_o);
        this->bindWeight(weights, this->name + ".b_c", & this->b_c);
        this->bindWeight(weights, this->name + ".b_q", & this->b_q);
    }
}

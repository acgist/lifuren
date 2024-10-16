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
    LFR_DEFINE_LAYER_2D(w_xi, input_size,  hidden_size)
    LFR_DEFINE_LAYER_2D(w_hi, hidden_size, hidden_size)
    // 遗忘门
    LFR_DEFINE_LAYER_2D(w_xf, input_size,  hidden_size)
    LFR_DEFINE_LAYER_2D(w_hf, hidden_size, hidden_size)
    // 输出门
    LFR_DEFINE_LAYER_2D(w_xo, input_size,  hidden_size)
    LFR_DEFINE_LAYER_2D(w_ho, hidden_size, hidden_size)
    // 候选记忆元
    LFR_DEFINE_LAYER_2D(w_xc, input_size,  hidden_size)
    LFR_DEFINE_LAYER_2D(w_hc, hidden_size, hidden_size)
    // 输出层
    LFR_DEFINE_LAYER_2D(w_hq, hidden_size, hidden_size)
    // 隐藏状态
    LFR_DEFINE_LAYER_2D(h, hidden_size, batch_size)
    LFR_DEFINE_LAYER_2D(c, hidden_size, batch_size)
    if(this->bias_) {
        // 输入门
        LFR_DEFINE_LAYER_1D(b_i, hidden_size)
        // 遗忘门
        LFR_DEFINE_LAYER_1D(b_f, hidden_size)
        // 输出门
        LFR_DEFINE_LAYER_1D(b_o, hidden_size)
        // 候选记忆元
        LFR_DEFINE_LAYER_1D(b_c, hidden_size)
        // 输出层
        LFR_DEFINE_LAYER_1D(b_q, hidden_size)
    }
}


void lifuren::layer::LSTM::bindWeight(const std::map<std::string, ggml_tensor*>& weights) {
    LFR_BIND_WEIGHT(w_xi)
    LFR_BIND_WEIGHT(w_hi)
    LFR_BIND_WEIGHT(w_xf)
    LFR_BIND_WEIGHT(w_hf)
    LFR_BIND_WEIGHT(w_xo)
    LFR_BIND_WEIGHT(w_ho)
    LFR_BIND_WEIGHT(w_xc)
    LFR_BIND_WEIGHT(w_hc)
    LFR_BIND_WEIGHT(w_hq)
    LFR_BIND_WEIGHT(h)
    LFR_BIND_WEIGHT(c)
    if(this->bias_) {
        LFR_BIND_WEIGHT(b_i)
        LFR_BIND_WEIGHT(b_f)
        LFR_BIND_WEIGHT(b_o)
        LFR_BIND_WEIGHT(b_c)
        LFR_BIND_WEIGHT(b_q)
    }
}

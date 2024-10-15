#include "lifuren/Layer.hpp"

#include "fmt/format.h"

lifuren::layer::Linear::Linear(
    size_t in_features,
    size_t out_features,
    ggml_context* ctx,
    const std::string& name,
    bool bias
) : Layer(ctx, ctx, name),
    in_features(in_features),
    out_features(out_features),
    bias_(bias)
{
}

lifuren::layer::Linear::Linear(
    size_t in_features,
    size_t out_features,
    ggml_context* ctx_weight,
    ggml_context* ctx_compute,
    const std::string& name, bool bias
) : Layer(ctx_weight, ctx_compute, name),
    in_features(in_features),
    out_features(out_features),
    bias_(bias)
{
}

lifuren::layer::Linear::~Linear() {
}

std::string lifuren::layer::Linear::info() const {
    return fmt::format("{} => in = {} out = {}", this->name, this->in_features, this->out_features);
}

ggml_tensor* lifuren::layer::Linear::forward(ggml_tensor* input) {
    ggml_tensor* mul_mat_ret = ggml_mul_mat(this->ctx_compute, this->weight, input);
    if(this->bias_) {
        return ggml_add(this->ctx_compute, mul_mat_ret, this->bias);
    }
    return mul_mat_ret;
}

void lifuren::layer::Linear::defineWeight() {
    this->weight = ggml_new_tensor_2d(this->ctx_weight, GGML_TYPE_F32, this->in_features, this->out_features);
    this->defineWeight(this->name + ".weight", this->weight);
    if(this->bias_) {
        this->bias = ggml_new_tensor_1d(this->ctx_weight, GGML_TYPE_F32, this->out_features);
        this->defineWeight(this->name + ".bias", this->bias);
    }
}

void lifuren::layer::Linear::bindWeight(const std::map<std::string, ggml_tensor*>& weights) {
    this->bindWeight(weights, this->name + ".weight", &this->weight);
    if(this->bias_) {
        this->bindWeight(weights, this->name + ".bias", &this->bias);
    }
}

#include "lifuren/Layers.hpp"

#include "ggml.h"

lifuren::layers::Linear::Linear(size_t in_features, size_t out_features, ggml_context* ctx, const std::string& name, bool bias) : in_features(in_features), out_features(out_features), bias_(bias), Layer(ctx, ctx, name) {
}

lifuren::layers::Linear::Linear(size_t in_features, size_t out_features, ggml_context* ctx_weight, ggml_context* ctx_compute, const std::string& name, bool bias) : in_features(in_features), out_features(out_features), bias_(bias), Layer(ctx_weight, ctx_compute, name) {
}

lifuren::layers::Linear::~Linear() {
}

std::string lifuren::layers::Linear::info() {
    float* w = ggml_get_data_f32(this->weight);
    if(this->bias_) {
        float* b = ggml_get_data_f32(this->bias);
        return this->name + " => w = " + std::to_string(*w) + " + b = " + std::to_string(*b);
    } else {
        return this->name + " => w = " + std::to_string(*w);
    }
}

ggml_tensor* lifuren::layers::Linear::forward(ggml_tensor* input) {
    if(this->bias_) {
        return ggml_add(this->ctx_compute, ggml_mul_mat(this->ctx_compute, this->weight, input), this->bias);
    } else {
        return ggml_mul_mat(this->ctx_compute, this->weight, input);
    }
}

void lifuren::layers::Linear::defineWeight(std::map<std::string, ggml_tensor*>& weights) {
    this->weight = ggml_new_tensor_2d(this->ctx_weight, GGML_TYPE_F32, this->in_features, this->out_features);
    weights.emplace(this->name + ".weight", this->weight);
    if(this->bias_) {
        this->bias = ggml_new_tensor_1d(this->ctx_weight, GGML_TYPE_F32, this->out_features);
        weights.emplace(this->name + ".bias", this->bias);
    }
}

void lifuren::layers::Linear::bindWeight(std::map<std::string, ggml_tensor*>& weights) {
    this->bindWeight(weights, this->name + ".weight", &this->weight);
    if(this->bias_) {
        this->bindWeight(weights, this->name + ".bias", &this->bias);
    }
}

#include "lifuren/Layers.hpp"

#include "ggml.h"

lifuren::layers::Conv2d::Conv2d(
    size_t in_channels,
    size_t out_channels,
    size_t kernel_size,
    ggml_context* ctx,
    const std::string& name,
    size_t stride,
    size_t padding,
    size_t dilation,
    bool   bias
) : in_channels(in_channels),
    out_channels(out_channels),
    kernel_size(kernel_size),
    stride(stride),
    padding(padding),
    dilation(dilation),
    bias_(bias),
    Layer(ctx, ctx, name) {
}

lifuren::layers::Conv2d::Conv2d(
    size_t in_channels,
    size_t out_channels,
    size_t kernel_size,
    ggml_context* ctx_weight,
    ggml_context* ctx_compute,
    const std::string& name,
    size_t stride,
    size_t padding,
    size_t dilation,
    bool   bias
) : in_channels(in_channels),
    out_channels(out_channels),
    kernel_size(kernel_size),
    stride(stride),
    padding(padding),
    dilation(dilation),
    bias_(bias),
    Layer(ctx_weight, ctx_compute, name) {
}

lifuren::layers::Conv2d::~Conv2d() {
}

std::string lifuren::layers::Conv2d::info() {
    return this->name + " => in = " + std::to_string(this->in_channels) + " out = " + std::to_string(this->out_channels) + " k = " + std::to_string(this->kernel_size);
}

ggml_tensor* lifuren::layers::Conv2d::forward(ggml_tensor* input) {
    ggml_tensor* conv_2d_ret = ggml_conv_2d(this->ctx_compute, this->kernel, input, this->stride, this->stride, this->padding, this->padding, this->dilation, this->dilation);
    if(this->bias_) {
        return ggml_add(this->ctx_compute, conv_2d_ret, this->bias);
    }
    return conv_2d_ret;
}

void lifuren::layers::Conv2d::defineWeight(std::map<std::string, ggml_tensor*>& weights) {
    this->kernel = ggml_new_tensor_4d(this->ctx_compute, GGML_TYPE_F32, this->kernel_size, this->kernel_size, this->in_channels, this->out_channels);
    weights.emplace(this->name + ".kernel", this->kernel);
    if(this->bias_) {
        // TODO: 偏置1还是计算w*h
        this->bias = ggml_new_tensor_3d(this->ctx_compute, GGML_TYPE_F32, 1, 1, this->out_channels);
        weights.emplace(this->name + ".bias", this->bias);
    }
}

void lifuren::layers::Conv2d::bindWeight(std::map<std::string, ggml_tensor*>& weights) {
    this->bindWeight(weights, this->name + ".kernel", &this->kernel);
    if(this->bias_) {
        this->bindWeight(weights, this->name + ".bias", &this->bias);
    }
}

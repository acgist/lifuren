#include "lifuren/Layers.hpp"

#include "ggml.h"

#include "spdlog/spdlog.h"

lifuren::layers::Layer::Layer(ggml_context* ctx_weight, ggml_context* ctx_compute, const std::string& name) : ctx_weight(ctx_weight), ctx_compute(ctx_compute), name(name) {
}

lifuren::layers::Layer::~Layer() {
}

std::string lifuren::layers::Layer::info() {
    return this->name;
}

ggml_tensor* lifuren::layers::Layer::operator()(ggml_tensor* input) {
    return this->forward(input);
}

void lifuren::layers::Layer::bindWeight(std::map<std::string, ggml_tensor*>& weights, const std::string& key, ggml_tensor** tensor) {
    auto iterator = weights.find(key);
    if(iterator == weights.end()) {
        SPDLOG_WARN("绑定权重失败：{}", key);
        return;
    }
    *tensor = iterator->second;
}

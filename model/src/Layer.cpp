#include "lifuren/Layer.hpp"

#include "spdlog/spdlog.h"

lifuren::layer::Layer::Layer(
    ggml_context* ctx_weight,
    ggml_context* ctx_compute,
    const std::string& name
) : ctx_weight(ctx_weight),
    ctx_compute(ctx_compute),
    name(name)
{
}

lifuren::layer::Layer::~Layer() {
}

std::string lifuren::layer::Layer::info() const {
    return this->name;
}

ggml_tensor* lifuren::layer::Layer::operator()(ggml_tensor* input) {
    return this->forward(input);
}

ggml_tensor* lifuren::layer::Layer::operator[](const char* name) {
    return ggml_get_tensor(this->ctx_weight, name);
}

void lifuren::layer::Layer::initWeight(std::function<void(ggml_tensor*)> function) {
    auto weight = ggml_get_first_tensor(this->ctx_weight);
    while(weight) {
        function(weight);
        weight = ggml_get_next_tensor(this->ctx_weight, weight);
    }
}

void lifuren::layer::Layer::defineWeight(const std::string& name, ggml_tensor* weight) const {
    lifuren::layer::defineWeight(name.c_str(), weight, this->ctx_compute);
}

 void lifuren::layer::Layer::bindWeight(const std::map<std::string, ggml_tensor*>& weights, const std::string& name, ggml_tensor** tensor) {
    const auto iterator = weights.find(name);
    if(iterator == weights.end()) {
        SPDLOG_WARN("权重无效：{}", name);
        return;
    }
    *tensor = iterator->second;
 }

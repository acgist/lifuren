#include "lifuren/EmbeddingClient.hpp"

#include "spdlog/spdlog.h"

#include "lifuren/Config.hpp"

lifuren::EmbeddingClient::EmbeddingClient(const std::string& path) : path(path) {
}

lifuren::EmbeddingClient::~EmbeddingClient() {
}

std::unique_ptr<lifuren::EmbeddingClient> lifuren::EmbeddingClient::getClient(const std::string& path, const std::string& embedding) {
    if(embedding == "ollama" || embedding == "Ollama") {
        return std::make_unique<lifuren::OllamaEmbeddingClient>(path);
    } else if (embedding == "pepper" || embedding == "Pepper") {
        return std::make_unique<lifuren::PepperEmbeddingClient>(path);
    } else {
        return nullptr;
    }
}

/**
 * 1. 累加法
 * 2. 平均法
 * 3. ISF嵌入法
 * 4. TF-IDF加权平均法
 */
std::vector<float> lifuren::EmbeddingClient::getVector(const std::vector<std::string>& prompts) const {
    if(prompts.empty()) {
        return {};
    }
    const size_t dims = this->getDims();
    std::vector<float> ret;
    ret.reserve(dims);
    for(const auto& prompt : prompts) {
        const std::vector<float> vec = this->getVector(prompt);
        if(vec.empty()) {
            SPDLOG_WARN("没有嵌入向量：{}", prompt);
            continue;
        }
        for(size_t i = 0; i < dims; ++i) {
            ret[i] += vec[i];
        }
    }
    const size_t size = prompts.size();
    for(size_t i = 0; i < dims; ++i) {
        ret[i] /= size;
    }
    return ret;
}

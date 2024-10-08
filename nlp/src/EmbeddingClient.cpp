#include "lifuren/EmbeddingClient.hpp"

#include "spdlog/spdlog.h"

lifuren::EmbeddingClient::EmbeddingClient() {
}

lifuren::EmbeddingClient::~EmbeddingClient() {
}

std::unique_ptr<lifuren::EmbeddingClient> lifuren::EmbeddingClient::getClient(const std::string& embedding) {
    if(embedding == "ollama" || embedding == "Ollama") {
        return std::make_unique<lifuren::OllamaEmbeddingClient>();
    } else if (
        embedding == "chinesewordvectors" || embedding == "chinese-word-vectors" ||
        embedding == "ChineseWordVectors" || embedding == "Chinese-Word-Vectors"
    ) {
        return std::make_unique<lifuren::ChineseWordVectorsEmbeddingClient>();
    } else {
        return nullptr;
    }
}

/**
 * 1. 累加法
 * 2. 平均法
 * 3. TF-IDF加权平均法
 * 4. ISF嵌入法
 */
std::vector<float> lifuren::EmbeddingClient::getVector(const std::vector<std::string>& prompts) const {
    if(prompts.empty()) {
        return {};
    }
    const size_t dims = this->getDims();
    std::vector<float> ret;
    ret.reserve(dims);
    for(const auto& prompt : prompts) {
        const std::vector<float>&& vec = this->getVector(prompt);
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

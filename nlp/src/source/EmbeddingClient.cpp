#include "lifuren/EmbeddingClient.hpp"

#include "spdlog/spdlog.h"

#include "lifuren/Poetrys.hpp"

lifuren::EmbeddingClient::EmbeddingClient() {
}

lifuren::EmbeddingClient::~EmbeddingClient() {
}

std::unique_ptr<lifuren::EmbeddingClient> lifuren::EmbeddingClient::getClient(const std::string& embedding) {
    if(embedding == "ollama") {
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
std::vector<float> lifuren::EmbeddingClient::getSegmentVector(const std::vector<std::string>& segment) const {
    std::map<std::string, std::vector<float>>&& ret = this->getVector(segment);
    if(ret.empty()) {
        return {};
    }
    std::vector<float>& head = ret.begin()->second;
    const size_t rows = ret.size();
    const size_t size = head.size();
    std::vector<float> data;
    data.resize(size);
    for(const auto& [key, value] : ret) {
        if(value.empty()) {
            SPDLOG_DEBUG("没有嵌入向量：{}", key);
            continue;
        }
        for(size_t i = 0; i < size; ++i) {
            data[i] += value[i];
        }
    }
    for(size_t i = 0; i < size; ++i) {
        data[i] /= rows;
    }
    return data;
}

std::map<std::string, std::vector<float>> lifuren::EmbeddingClient::getVector(const std::vector<std::string>& segment) const {
    if(segment.empty()) {
        return {};
    }
    std::map<std::string, std::vector<float>> ret;
    for(const auto& word : segment) {
        ret.emplace(word, std::move(this->getVector(word)));
    }
    return ret;
}

bool lifuren::EmbeddingClient::release() {
    return true;
}

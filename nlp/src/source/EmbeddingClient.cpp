#include "lifuren/EmbeddingClient.hpp"

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

std::vector<std::vector<float>> lifuren::EmbeddingClient::getVector(const std::vector<std::string>& segment) {
    if(segment.empty()) {
        return {};
    }
    std::vector<std::vector<float>> ret;
    ret.reserve(segment.size());
    for(const auto& word : segment) {
        ret.emplace_back(std::move(this->getVector(word)));
    }
    return std::move(ret);
}

/**
 * 1. 累加法
 * 2. 平均法
 * 3. TF-IDF加权平均法
 * 4. ISF嵌入法
 */
std::vector<float> lifuren::EmbeddingClient::getSegmentVector(const std::vector<std::string>& segment) {
    std::vector<std::vector<float>>&& ret = this->getVector(segment);
    if(ret.empty()) {
        return {};
    }
    std::vector<float>& head = ret[0];
    const size_t rows = ret.size();
    const size_t size = head.size();
    std::vector<float> data;
    data.reserve(size);
    for(const auto& v : ret) {
        for(size_t i = 0; i < size; ++i) {
            data[i] += v[i];
        }
    }
    for(size_t i = 0; i < size; ++i) {
        data[i] /= rows;
    }
    return data;
}

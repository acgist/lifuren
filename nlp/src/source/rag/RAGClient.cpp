#include "lifuren/RAG.hpp"

#include "spdlog/spdlog.h"

lifuren::RAGClient::RAGClient(size_t id, const std::string& path, const std::string& embedding) : id(id), path(path) {
    this->embeddingService = std::make_unique<lifuren::EmbeddingService>(embedding);
}

lifuren::RAGClient::~RAGClient() {
}

std::unique_ptr<lifuren::RAGClient> lifuren::RAGClient::getRAGClient(const std::string& type, size_t id, const std::string& path, const std::string& embedding) {
    if(type == "elasticsearch" || type == "ElasticSearch") {
        return std::make_unique<lifuren::ElasticSearchRAGClient>(id, path, embedding);
    } else {
        SPDLOG_WARN("不支持的RAGClient类型：{}", type);
    }
    return nullptr;
}

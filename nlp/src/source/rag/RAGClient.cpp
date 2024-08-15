#include "lifuren/RAG.hpp"

lifuren::RAGClient::RAGClient(const std::string& embedding) {
    this->embeddingService = std::make_unique<lifuren::EmbeddingService>(embedding);
}

lifuren::RAGClient::~RAGClient() {
}

#include "lifuren/RAG.hpp"

lifuren::EmbeddingService::EmbeddingService(const std::string& embedding) {
    if(embedding == "ollama") {
        this->embeddingClient = std::make_unique<lifuren::OllamaEmbeddingClient>();
    } else {
    }
}

lifuren::EmbeddingService::~EmbeddingService() {
}

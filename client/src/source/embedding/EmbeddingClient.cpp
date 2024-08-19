#include "lifuren/Client.hpp"

std::unique_ptr<lifuren::EmbeddingClient> lifuren::EmbeddingClient::getClient(const std::string& embedding) {
    if(embedding == "ollama") {
        return std::make_unique<lifuren::OllamaEmbeddingClient>();
    } else {
        return nullptr;
    }
}
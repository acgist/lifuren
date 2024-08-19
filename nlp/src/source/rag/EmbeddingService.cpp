#include "lifuren/RAG.hpp"

lifuren::EmbeddingService::EmbeddingService(const std::string& embedding) : embeddingClient(lifuren::EmbeddingClient::getClient(embedding)) {
}

lifuren::EmbeddingService::~EmbeddingService() {
}

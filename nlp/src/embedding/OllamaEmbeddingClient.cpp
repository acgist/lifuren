/**
 * https://github.com/ollama/ollama/blob/main/docs/api.md
 */
#include "lifuren/EmbeddingClient.hpp"

#include "spdlog/spdlog.h"

#include "nlohmann/json.hpp"

lifuren::OllamaEmbeddingClient::OllamaEmbeddingClient() : EmbeddingClient() {
    const auto& ollamaConfig = lifuren::config::CONFIG.ollama;
    this->restClient = std::make_unique<lifuren::RestClient>(ollamaConfig.api);
    this->restClient->auth(ollamaConfig);
}

lifuren::OllamaEmbeddingClient::~OllamaEmbeddingClient() {
}

std::vector<float> lifuren::OllamaEmbeddingClient::getVector(const std::string& prompt) const {
    const auto& ollamaConfig    = lifuren::config::CONFIG.ollama;
    const auto& embeddingConfig = ollamaConfig.embeddingClient;
    nlohmann::json body = {
        { "model", embeddingConfig.model },
        { "input", prompt                }
    };
    const auto&& response = this->restClient->postJson(embeddingConfig.path, body.dump());
    if(!response) {
        return {};
    }
    nlohmann::json data = nlohmann::json::parse(response.body);
    auto iterator = data.find("embeddings");
    if(iterator == data.end()) {
        return {};
    }
    auto embeddings = iterator->get<std::vector<std::vector<float>>>();
    if(embeddings.size() == 0LL || embeddings.size() > 1LL) {
        SPDLOG_WARN("Ollama词嵌入返回错误：{}", prompt);
        return {};
    }
    return embeddings[0];
}

size_t lifuren::OllamaEmbeddingClient::getDims() const {
    return 1024;
}

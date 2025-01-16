#include "lifuren/EmbeddingClient.hpp"

#include "spdlog/spdlog.h"

#include "nlohmann/json.hpp"

#include "lifuren/Client.hpp"

lifuren::OllamaEmbeddingClient::OllamaEmbeddingClient(const std::string& path) : EmbeddingClient(path) {
    const auto& ollamaConfig = lifuren::config::CONFIG.ollama;
    this->restClient = std::make_unique<lifuren::RestClient>(ollamaConfig.api);
    this->restClient->auth(ollamaConfig);
}

lifuren::OllamaEmbeddingClient::~OllamaEmbeddingClient() {
}

std::vector<float> lifuren::OllamaEmbeddingClient::getVector(const std::string& prompt) const {
    const auto& ollamaConfig = lifuren::config::CONFIG.ollama;
    nlohmann::json body = {
        { "model", ollamaConfig.model },
        { "input", prompt             }
    };
    const auto response = std::move(this->restClient->postJson(ollamaConfig.path, body.dump()));
    if(!response) {
        return {};
    }
    nlohmann::json data = std::move(nlohmann::json::parse(response.body));
    auto iterator = data.find("embeddings");
    if(iterator == data.end()) {
        SPDLOG_WARN("没有匹配嵌入内容：{}", prompt);
        return {};
    }
    const auto embeddings = std::move(iterator->get<std::vector<std::vector<float>>>());
    if(embeddings.size() != 1) {
        SPDLOG_WARN("Ollama词嵌入返回错误：{}", prompt);
        return {};
    }
    return embeddings[0];
}

size_t lifuren::OllamaEmbeddingClient::getDims() const {
    return lifuren::config::CONFIG.ollama.dims;
}

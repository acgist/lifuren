/**
 * https://github.com/ollama/ollama/blob/main/docs/api.md
 */
#include "lifuren/EmbeddingClient.hpp"

#include "nlohmann/json.hpp"

lifuren::OllamaEmbeddingClient::OllamaEmbeddingClient() {
    const auto& ollamaConfig = lifuren::config::CONFIG.ollama;
    this->restClient = std::make_unique<lifuren::RestClient>(ollamaConfig.api);
    this->restClient->auth(ollamaConfig);
}

lifuren::OllamaEmbeddingClient::~OllamaEmbeddingClient() {
}

std::vector<float> lifuren::OllamaEmbeddingClient::getVector(const std::string& word) {
    const auto& ollamaConfig    = lifuren::config::CONFIG.ollama;
    const auto& embeddingConfig = ollamaConfig.embeddingClient;
    nlohmann::json body = {
        { "model", embeddingConfig.model },
        { "input", word }
    };
    const auto&& response = this->restClient->postJson(embeddingConfig.path, body.dump());
    if(!response.success) {
        return {};
    }
    nlohmann::json data = nlohmann::json::parse(response.body);
    std::vector<float> ret{};
    if(data.contains("embeddings")) {
        const auto& embeddings = data["embeddings"].get<std::vector<std::vector<float>>>();
        for(const auto& embedding : embeddings) {
            ret.reserve(ret.size() + embedding.size());
            ret.insert(ret.begin(), embedding.begin(), embedding.end());
        }
    }
    return ret;
}

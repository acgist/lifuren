#include "lifuren/String.hpp"

#include "lifuren/EmbeddingClient.hpp"

std::vector<float> lifuren::string::embedding(const std::vector<std::string>& prompts) {
    auto client = lifuren::EmbeddingClient::getClient(lifuren::config::CONFIG_OLLAMA);
    return std::move(client->getVector(prompts));
}

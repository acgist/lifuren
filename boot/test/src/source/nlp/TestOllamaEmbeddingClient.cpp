#include "Test.hpp"
#include "lifuren/EmbeddingClient.hpp"

static void testEmbedding() {
    lifuren::OllamaEmbeddingClient client{};
    const auto&& vector = client.getVector("李夫人");
    SPDLOG_DEBUG("v length = {}", vector.size());
}

LFR_TEST(
    testEmbedding();
);

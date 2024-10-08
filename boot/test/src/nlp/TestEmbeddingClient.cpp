#include "Test.hpp"

#include "lifuren/EmbeddingClient.hpp"

[[maybe_unused]] static void testOllamaEmbedding() {
    lifuren::OllamaEmbeddingClient client{};
    const auto&& vector = client.getVector("李夫人");
    SPDLOG_DEBUG("v length = {}", vector.size());
}

[[maybe_unused]] static void testChineseWordVectorsEmbedding() {
    lifuren::ChineseWordVectorsEmbeddingClient client{};
    auto v = client.getVector("中");
    // auto v = client.getVector({ "李", "夫", "人"});
    SPDLOG_DEBUG("v length = {}", v.size());
}

LFR_TEST(
    // testOllamaEmbedding();
    testChineseWordVectorsEmbedding();
);

#include "lifuren/Test.hpp"

#include "lifuren/EmbeddingClient.hpp"

[[maybe_unused]] static void testOllamaEmbedding() {
    lifuren::OllamaEmbeddingClient client{};
    const auto v = std::move(client.getVector("李夫人"));
    SPDLOG_DEBUG("v length = {}", v.size());
}

[[maybe_unused]] static void testChineseWordVectorsEmbedding() {
    // lifuren::ChineseWordVectorsEmbeddingClient ref{};
    {
        lifuren::ChineseWordVectorsEmbeddingClient client{};
        auto v = std::move(client.getVector("中"));
        // auto v = std::move(client.getVector({ "李", "夫", "人"}));
        SPDLOG_DEBUG("v length = {}", v.size());
    }
    SPDLOG_DEBUG("释放1");
    SPDLOG_DEBUG("释放2");
    SPDLOG_DEBUG("释放3");
}

LFR_TEST(
    // testOllamaEmbedding();
    testChineseWordVectorsEmbedding();
);

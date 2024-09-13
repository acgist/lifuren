#include "Test.hpp"

#include "lifuren/EmbeddingClient.hpp"

[[maybe_unused]] static void testEmbedding() {
    lifuren::ChineseWordVectorsEmbeddingClient client{};
    auto v = client.getVector("中");
    SPDLOG_DEBUG("v = {}", v.size());
}

[[maybe_unused]] static void testRelease() {
    lifuren::ChineseWordVectorsEmbeddingClient client{};
    client.release();
}

LFR_TEST(
    testEmbedding();
    testRelease();
);

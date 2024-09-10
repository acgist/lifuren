#include "Test.hpp"
#include "lifuren/EmbeddingClient.hpp"

static void testEmbedding() {
    lifuren::ChineseWordVectorsEmbeddingClient client{};
    auto v = client.getVector("中");
    SPDLOG_DEBUG("v = {}", v.size());
}

static void testRelease() {
    lifuren::ChineseWordVectorsEmbeddingClient client{};
    client.release();
}

LFR_TEST(
    testEmbedding();
    testRelease();
);

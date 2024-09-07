#include "lifuren/EmbeddingClient.hpp"

#include "spdlog/spdlog.h"

#include "lifuren/Logger.hpp"

static void testEmbedding() {
    lifuren::ChineseWordVectorsEmbeddingClient client{};
    auto v = client.getVector("中");
    SPDLOG_DEBUG("v = {}", v.size());
}

static void testRelease() {
    lifuren::ChineseWordVectorsEmbeddingClient client{};
    client.release();
}

int main() {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    testEmbedding();
    testRelease();
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}

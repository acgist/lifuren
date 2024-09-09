#include "lifuren/EmbeddingClient.hpp"

#include "spdlog/spdlog.h"

#include "lifuren/Logger.hpp"

static void testEmbedding() {
    lifuren::OllamaEmbeddingClient client{};
    const auto&& vector = client.getVector("李夫人");
    SPDLOG_DEBUG("v length = {}", vector.size());
}

int main() {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    testEmbedding();
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}
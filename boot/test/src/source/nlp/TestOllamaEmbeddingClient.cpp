#include "lifuren/EmbeddingClient.hpp"

#include "spdlog/spdlog.h"

#include "lifuren/Logger.hpp"

static void testEmbedding() {
    lifuren::OllamaEmbeddingClient client{};
    const auto&& vector = client.getVector("李夫人");
}

int main() {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}
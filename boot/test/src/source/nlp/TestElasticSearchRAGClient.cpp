#include "lifuren/RAGClient.hpp"

#include "spdlog/spdlog.h"

#include "lifuren/Logger.hpp"

static void testRAGClient() {
    lifuren::ElasticSearchRAGClient client{ "D:/tmp/test", "ollama" };
    client.index("1234");
}

int main() {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    testRAGClient();
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}
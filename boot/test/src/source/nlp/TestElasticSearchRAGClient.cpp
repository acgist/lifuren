#include "lifuren/RAGClient.hpp"

#include "spdlog/spdlog.h"

#include "lifuren/Logger.hpp"

static void testRAGClient() {
    lifuren::ElasticSearchRAGClient client{ "D:/tmp/test", "ollama" };
    // lifuren::ElasticSearchRAGClient client{ "D:/tmp/test", "ChineseWordVectors" };
    client.loadIndex();
    client.deleteRAG();
    client.index("猪");
    client.index("牛");
    client.index("马");
    client.index("桃子");
    client.index("苹果");
    client.index("李子");
    auto a = client.search("狗");
    for(const auto& v : a) {
        SPDLOG_DEBUG("狗 = {}", v);
    }
    auto b = client.search("水果");
    // auto b = client.search("草莓");
    for(const auto& v : b) {
        SPDLOG_DEBUG("水果 = {}", v);
        // SPDLOG_DEBUG("草莓 = {}", v);
    }
    client.saveIndex();
}

int main() {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    testRAGClient();
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}
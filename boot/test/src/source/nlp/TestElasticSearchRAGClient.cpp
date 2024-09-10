#include "Test.hpp"
#include "lifuren/RAG.hpp"

#include <chrono>
#include <thread>

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
    // 这里需要等待索引建立
    std::this_thread::sleep_for(std::chrono::seconds(2));
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

LFR_TEST(
    testRAGClient();
);

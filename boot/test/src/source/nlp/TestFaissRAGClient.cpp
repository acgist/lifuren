#include "Test.hpp"

#include "lifuren/RAG.hpp"

[[maybe_unused]] static void testRAGClient() {
    lifuren::FaissRAGClient client{ "D:/tmp/test", "ollama" };
    client.loadIndex();
    client.deleteRAG();
    client.index("猪");
    client.index("牛");
    client.index("马");
    client.index("桃子");
    client.index("苹果");
    client.index("李子");
    auto&& a = client.search("水果");
    for(const auto& v : a) {
        SPDLOG_DEBUG("水果 = {}", v);
    }
    auto&& b = client.search("动物");
    for(const auto& v : b) {
        SPDLOG_DEBUG("动物 = {}", v);
    }
    client.saveIndex();
}

LFR_TEST(
    testRAGClient();
);

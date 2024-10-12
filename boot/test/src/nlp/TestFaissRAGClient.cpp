#include "lifuren/Test.hpp"

#include "lifuren/RAG.hpp"
#include "lifuren/File.hpp"

[[maybe_unused]] static void testRAGClientIndex() {
    lifuren::FaissRAGClient client{ lifuren::file::join({lifuren::config::CONFIG.tmp, "docs"}).string(), "ollama" };
    client.loadIndex();
    client.index("猪");
    client.index("牛");
    client.index("马");
    client.index("桃子");
    client.index("苹果");
    client.index("李子");
    client.saveIndex();
}

[[maybe_unused]] static void testRAGClientSearch() {
    lifuren::FaissRAGClient client{ lifuren::file::join({lifuren::config::CONFIG.tmp, "docs"}).string(), "ollama" };
    client.loadIndex();
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

[[maybe_unused]] static void testRAGClientTruncate() {
    lifuren::FaissRAGClient client{ lifuren::file::join({lifuren::config::CONFIG.tmp, "docs"}).string(), "ollama" };
    client.loadIndex();
    client.truncateIndex();
}

LFR_TEST(
    // testRAGClientIndex();
    testRAGClientSearch();
    // testRAGClientTruncate();
);

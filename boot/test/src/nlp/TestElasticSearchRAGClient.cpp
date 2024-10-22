#include "lifuren/Test.hpp"

#include "lifuren/RAG.hpp"
#include "lifuren/File.hpp"

[[maybe_unused]] static void testRAGClientIndex() {
    lifuren::ElasticSearchRAGClient client{ lifuren::file::join({lifuren::config::CONFIG.tmp, "docs"}).string(), "ollama" };
    // lifuren::ElasticSearchRAGClient client{ lifuren::file::join({lifuren::config::CONFIG.tmp, "docs"}).string(), "ChineseWordVectors" };
    client.loadIndex();
    client.index("猪");
    client.index("牛");
    client.index("马");
    client.index("马");
    client.index("马");
    client.index("桃子");
    client.index("桃子");
    client.index("桃子");
    client.index("苹果");
    client.index("李子");
    client.saveIndex();
}

[[maybe_unused]] static void testRAGClientSearch() {
    lifuren::ElasticSearchRAGClient client{ lifuren::file::join({lifuren::config::CONFIG.tmp, "docs"}).string(), "ollama" };
    // lifuren::ElasticSearchRAGClient client{ lifuren::file::join({lifuren::config::CONFIG.tmp, "docs"}).string(), "ChineseWordVectors" };
    client.loadIndex();
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

[[maybe_unused]] static void testRAGClientTruncate() {
    lifuren::ElasticSearchRAGClient client{ lifuren::file::join({lifuren::config::CONFIG.tmp, "docs"}).string(), "ollama" };
    client.loadIndex();
    client.truncateIndex();
}

LFR_TEST(
    // testRAGClientIndex();
    testRAGClientSearch();
    // testRAGClientTruncate();
);

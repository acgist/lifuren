#include "lifuren/Test.hpp"

#include "lifuren/File.hpp"
#include "lifuren/RAGClient.hpp"

[[maybe_unused]] static void testRAGClientIndex() {
    lifuren::FaissRAGClient client{ lifuren::file::join({lifuren::config::CONFIG.tmp, "docs"}).string(), "ollama" };
    // lifuren::FaissRAGClient client{ lifuren::file::join({lifuren::config::CONFIG.tmp, "docs"}).string(), "pepper" };
    // lifuren::ElasticSearchRAGClient client{ lifuren::file::join({lifuren::config::CONFIG.tmp, "docs"}).string(), "ollama" };
    // lifuren::ElasticSearchRAGClient client{ lifuren::file::join({lifuren::config::CONFIG.tmp, "docs"}).string(), "pepper" };
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
    // lifuren::FaissRAGClient client{ lifuren::file::join({lifuren::config::CONFIG.tmp, "docs"}).string(), "ollama" };
    // lifuren::FaissRAGClient client{ lifuren::file::join({lifuren::config::CONFIG.tmp, "docs"}).string(), "pepper" };
    lifuren::ElasticSearchRAGClient client{ lifuren::file::join({lifuren::config::CONFIG.tmp, "docs"}).string(), "ollama" };
    // lifuren::ElasticSearchRAGClient client{ lifuren::file::join({lifuren::config::CONFIG.tmp, "docs"}).string(), "pepper" };
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
    // lifuren::FaissRAGClient client{ lifuren::file::join({lifuren::config::CONFIG.tmp, "docs"}).string(), "ollama" };
    lifuren::ElasticSearchRAGClient client{ lifuren::file::join({lifuren::config::CONFIG.tmp, "docs"}).string(), "ollama" };
    client.loadIndex();
    client.truncateIndex();
}

[[maybe_unused]] static void testRAGClientRag() {
    lifuren::RAGClient::rag("faiss", lifuren::file::join({lifuren::config::CONFIG.tmp, "docs"}).string(), "ollama");
}

LFR_TEST(
    testRAGClientIndex();
    // testRAGClientSearch();
    // testRAGClientTruncate();
    testRAGClientRag();
);

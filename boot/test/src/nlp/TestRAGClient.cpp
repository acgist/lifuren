#include "lifuren/Test.hpp"

#include "lifuren/File.hpp"
#include "lifuren/Dataset.hpp"
#include "lifuren/RAGClient.hpp"

[[maybe_unused]] static void testRAGClientIndex() {
    lifuren::FaissRAGClient client{ lifuren::file::join({lifuren::config::CONFIG.tmp, "docs"}).string(), "ollama" };
    // lifuren::FaissRAGClient client{ lifuren::file::join({lifuren::config::CONFIG.tmp, "docs"}).string(), "pepper" };
    // lifuren::ElasticSearchRAGClient client{ lifuren::file::join({lifuren::config::CONFIG.tmp, "docs"}).string(), "ollama" };
    // lifuren::ElasticSearchRAGClient client{ lifuren::file::join({lifuren::config::CONFIG.tmp, "docs"}).string(), "pepper" };
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
}

[[maybe_unused]] static void testRAGClientSearch() {
    lifuren::FaissRAGClient client{ lifuren::file::join({lifuren::config::CONFIG.tmp, "docs"}).string(), "ollama" };
    // lifuren::FaissRAGClient client{ lifuren::file::join({lifuren::config::CONFIG.tmp, "docs"}).string(), "pepper" };
    // lifuren::ElasticSearchRAGClient client{ lifuren::file::join({lifuren::config::CONFIG.tmp, "docs"}).string(), "ollama" };
    // lifuren::ElasticSearchRAGClient client{ lifuren::file::join({lifuren::config::CONFIG.tmp, "docs"}).string(), "pepper" };
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
}

[[maybe_unused]] static void testRAGEmbedding() {
    const std::string rag       = "faiss";
    const std::string path      = lifuren::file::join({ lifuren::config::CONFIG.tmp, "poetry-embedding" }).string();
    const std::string embedding = "pepper";
    std::shared_ptr<lifuren::RAGClient> client = std::move(lifuren::RAGClient::getClient(rag, path, embedding));
    auto embeddingFunction = std::bind(&lifuren::poetry::ragEmbedding, client, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);
    lifuren::dataset::allDatasetPreprocessing(path, lifuren::config::EMBEDDING_MODEL_FILE, embeddingFunction);
}

LFR_TEST(
    // testRAGClientIndex();
    testRAGClientSearch();
    // testRAGEmbedding();
);

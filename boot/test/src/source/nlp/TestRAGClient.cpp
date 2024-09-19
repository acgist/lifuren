#include "Test.hpp"

#include <thread>
#include <chrono>

#include "lifuren/RAG.hpp"
#include "lifuren/Files.hpp"

[[maybe_unused]] static void testRAGService() {
    auto& service = lifuren::RAGService::getInstance();
    auto ptr = service.runRAGTask(lifuren::files::join({lifuren::config::CONFIG.tmp, "docs"}).string());
    while(!ptr->finish) {
        SPDLOG_DEBUG("当前进度：{}", ptr->percent());
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    SPDLOG_DEBUG("任务完成");
}

[[maybe_unused]] static void testRAGTaskRunner() {
    lifuren::RAGTask task {
        .type      = "elasticsearch",
        .embedding = "text",
        .path      = lifuren::files::join({lifuren::config::CONFIG.tmp, "docs"}).string(),
    };
    lifuren::RAGTaskRunner runner{ task };
    std::this_thread::sleep_for(std::chrono::seconds(16));
}

[[maybe_unused]] static void testElasticSearchRAGClient() {
    lifuren::ElasticSearchRAGClient client{ lifuren::files::join({lifuren::config::CONFIG.tmp, "docs"}).string(), "" };
    // client.index("李夫人");
    client.search("李夫人");
    // client.truncateIndex();
}

LFR_TEST(
    testRAGService();
    // testRAGTaskRunner();
    // testElasticSearchRAGClient();
);

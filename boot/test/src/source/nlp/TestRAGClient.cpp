#include "Test.hpp"

#include <thread>
#include <chrono>

#include "lifuren/RAG.hpp"

[[maybe_unused]] static void testRAGService() {
    auto& service = lifuren::RAGService::getInstance();
    auto ptr = service.buildRAGTask("D:/tmp/docs");
    while(!ptr->finish) {
        SPDLOG_DEBUG("当前进度：{}", ptr->percent());
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    SPDLOG_DEBUG("任务完成");
}

[[maybe_unused]] static void testRAGTaskRunner() {
    lifuren::RAGTask task {
        .type      = "elasticsearch",
        .path      = "D:/tmp/docs",
        .embedding = "text",
    };
    lifuren::RAGTaskRunner runner{ task };
    std::this_thread::sleep_for(std::chrono::seconds(16));
}

[[maybe_unused]] static void testElasticSearchRAGClient() {
    lifuren::ElasticSearchRAGClient client{ "D:/tmp/docs", "" };
    // client.index("李夫人");
    client.search("李夫人");
    // client.deleteRAG();
}

LFR_TEST(
    testRAGService();
    // testRAGTaskRunner();
    // testElasticSearchRAGClient();
);

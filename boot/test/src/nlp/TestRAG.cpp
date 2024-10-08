#include "Test.hpp"

#include <thread>
#include <chrono>

#include "lifuren/RAG.hpp"
#include "lifuren/Files.hpp"

[[maybe_unused]] static void testRAGTaskRunner() {
    lifuren::RAGTask task {
        .rag       = "elasticsearch",
        .embedding = "ollama",
        .path      = lifuren::files::join({lifuren::config::CONFIG.tmp, "docs"}).string(),
    };
    lifuren::RAGTaskRunner runner{ task };
}

[[maybe_unused]] static void testRAGService() {
    auto& service = lifuren::RAGService::getInstance();
    const auto ptr = service.runRAGTask(lifuren::files::join({lifuren::config::CONFIG.tmp, "docs"}).string());
    while(!ptr->finish) {
        SPDLOG_DEBUG("当前进度：{}", ptr->percent());
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    SPDLOG_DEBUG("任务完成");
    service.taskCount();
    service.stopRAGTask(lifuren::config::CONFIG.tmp);
    service.removeRAGTask(lifuren::config::CONFIG.tmp);
}

LFR_TEST(
    // testRAGTaskRunner();
    testRAGService();
);

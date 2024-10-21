#include "lifuren/Test.hpp"

#include <thread>
#include <chrono>

#include "lifuren/RAG.hpp"
#include "lifuren/File.hpp"

[[maybe_unused]] static void testRAGTaskRunner() {
    lifuren::RAGTask task{
        .rag       = "elasticsearch",
        .path      = lifuren::file::join({lifuren::config::CONFIG.tmp, "docs"}).string(),
        .embedding = "ollama",
    };
    lifuren::RAGTaskRunner runner{ task };
}

[[maybe_unused]] static void testRAGService() {
    auto& service = lifuren::RAGService::getInstance();
    const auto ptr = service.runRAGTask(lifuren::file::join({lifuren::config::CONFIG.tmp, "docs"}).string());
    while(!ptr->finish) {
        SPDLOG_DEBUG("当前进度：{}", ptr->percent());
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    SPDLOG_DEBUG("剩余任务数量：{}", service.taskCount());
    service.stopRAGTask(lifuren::config::CONFIG.tmp);
    service.removeRAGTask(lifuren::config::CONFIG.tmp);
}

LFR_TEST(
    // testRAGTaskRunner();
    testRAGService();
);

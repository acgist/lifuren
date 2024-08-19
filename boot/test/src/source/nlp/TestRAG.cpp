#include "lifuren/RAG.hpp"

#include <thread>
#include <chrono>

#include "spdlog/spdlog.h"

#include "lifuren/Logger.hpp"

[[maybe_unused]] static void testRAGService() {
    lifuren::RAGTask task {
        "D:/tmp/docs",
        "elasticsearch",
        "LINE",
        "elasticsearch"
    };
    auto& service = lifuren::RAGService::getInstance();
    auto ptr = service.buildRAGTask(task);
    while(!ptr->finish) {
        SPDLOG_DEBUG("当前进度：{}", ptr->percent());
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    SPDLOG_DEBUG("任务完成");
}

[[maybe_unused]] static void testRAGTaskRunner() {
    lifuren::RAGTask task {
        "D:/tmp/docs",
        "elasticsearch",
        "LINE",
        "elasticsearch"
    };
    lifuren::RAGTaskRunner runner{ task };
    std::this_thread::sleep_for(std::chrono::seconds(16));
}

[[maybe_unused]] static void testElasticSearchRAGClient() {
    lifuren::ElasticSearchRAGClient client{ "D:/tmp/docs", "" };
    // client.index("李夫人");
    client.search("李");
    // client.deleteRAG();
}

int main() {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    testRAGService();
    // testRAGTaskRunner();
    // testElasticSearchRAGClient();
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}
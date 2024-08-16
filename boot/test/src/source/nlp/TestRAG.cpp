#include "lifuren/RAG.hpp"

#include <thread>

#include "spdlog/spdlog.h"

#include "lifuren/Logger.hpp"

static void testRAGTaskRunner() {
    lifuren::RAGTask task {
        "D:/tmp/docs",
        "elasticsearch",
        "LINE",
        "elasticsearch"
    };
    lifuren::RAGTaskRunner runner{ task };
    std::this_thread::sleep_for(std::chrono::seconds::duration(120));
}

static void testElasticSearchRAGClient() {
    lifuren::ElasticSearchRAGClient client{ 8888, "D:/tmp/docs", "" };
    // client.index("李夫人");
    client.search("李");
    // client.deleteRAG();
}

int main() {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    testRAGTaskRunner();
    // testElasticSearchRAGClient();
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}
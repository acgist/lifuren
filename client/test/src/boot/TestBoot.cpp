#include "lifuren/Test.hpp"

#include "lifuren/Config.hpp"
#include "lifuren/Thread.hpp"

[[maybe_unused]] static void testUuid() {
    SPDLOG_DEBUG("uuid : {}", lifuren::config::uuid());
    SPDLOG_DEBUG("uuid : {}", lifuren::config::uuid());
}

[[maybe_unused]] static void testConfig() {
    auto config  = lifuren::config::CONFIG.loadFile();
    auto success = lifuren::config::CONFIG.saveFile();
    SPDLOG_DEBUG("wudaozi : {}", config.model_wudaozi);
    SPDLOG_DEBUG("shikuang: {}", config.model_shikuang);
    assert(success);
}

[[maybe_unused]] static void testThreadPool() {
    lifuren::thread::ThreadPool pool(4);
    for(int i = 0; i < 10; ++i) {
        pool.submit([i]() {
            SPDLOG_INFO("i : {}", i);
        });
    }
    pool.awaitTermination();
    SPDLOG_DEBUG("-");
}

LFR_TEST(
    testUuid();
    testConfig();
    testThreadPool();
);

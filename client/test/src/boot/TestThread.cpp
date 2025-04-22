#include "lifuren/Test.hpp"

#include "lifuren/Thread.hpp"

[[maybe_unused]] static void testThreadPool() {
    lifuren::thread::ThreadPool pool(4);
    for(int i = 0; i < 10; ++i) {
        pool.submit([i]() {
            SPDLOG_INFO("i : {}", i);
        });
    }
    pool.awaitTermination();
}

LFR_TEST(
    testThreadPool();
);

#include "lifuren/Test.hpp"

#include "lifuren/Thread.hpp"

[[maybe_unused]] static void testThreadPool() {
    lifuren::thread::ThreadPool pool(false, 4);
    for(int i = 0; i < 10; ++i) {
        pool.submit([i]() {
            SPDLOG_INFO("i : {}", i);
        });
    }
    pool.wait_finish();
}

[[maybe_unused]] static void testThreadTimer() {
    int time = 0;
    lifuren::thread::ThreadTimer timer;
    timer.schedule(2, [&time]() {
        SPDLOG_INFO("2 timer = {}", time++);
    });
    timer.schedule(5, [&time]() {
        SPDLOG_INFO("5 timer = {}", time++);
    });
    // std::this_thread::sleep_for(std::chrono::seconds(1));
    // timer.shutdown();
    std::this_thread::sleep_for(std::chrono::seconds(15));
    timer.shutdown();
}

LFR_TEST(
    // testThreadPool();
    testThreadTimer();
);

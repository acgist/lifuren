/**
 * 李夫人 - 玉簪花神
 * 
 * @author acgist
 */
#if __REST__
#include "REST.hpp"
#endif
#if __FLTK__
#include "FLTK.hpp"
#endif

#include <mutex>
#include <atomic>
#include <thread>
#include <condition_variable>

static std::mutex mutex;
static std::atomic<int> count(0);
static std::condition_variable condition;

int main(const int argc, const char * const argv[]) {
    lifuren::logger::init();
    SPDLOG_DEBUG("启动系统");
    #if __REST__
    count++;
    std::thread httpServerThread([]() {
        lifuren::initHttpServer();
        condition.notify_all();
    });
    httpServerThread.detach();
    #endif
    #if __FLTK__
    count++;
    std::thread fltkWindowThread([]() {
        lifuren::initFltkWindow();
        condition.notify_all();
    });
    fltkWindowThread.detach();
    #endif
    SPDLOG_DEBUG("启动完成");
    std::unique_lock lock(mutex);
    while(count > 0) {
        condition.wait(lock);
        count--;
    }
    SPDLOG_DEBUG("系统退出");
    lifuren::logger::shutdown();
    return 0;
}

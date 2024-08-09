/**
 * 李夫人 - 玉簪花神
 * 
 * @author acgist
 */
#if LFR_ENABLE_REST
#include "lifuren/REST.hpp"
#endif
#if LFR_ENABLE_FLTK
#include "lifuren/FLTK.hpp"
#endif

#include <mutex>
#include <atomic>
#include <thread>
#include <exception>
#include <condition_variable>

#include "spdlog/spdlog.h"

#include "lifuren/Logger.hpp"

static std::mutex mutex;
static std::atomic<int> count(0);
static std::condition_variable condition;

/**
 * 启动项目
 */
static void launch() {
    #if LFR_ENABLE_REST
    count++;
    std::thread httpServerThread([]() {
        lifuren::initHttpServer();
        {
            std::lock_guard<std::mutex> lock(mutex);
            count--;
            condition.notify_all();
        }
    });
    httpServerThread.detach();
    #endif
    #if LFR_ENABLE_FLTK
    count++;
    std::thread fltkWindowThread([]() {
        lifuren::initFltkWindow();
        {
            std::lock_guard<std::mutex> lock(mutex);
            count--;
            condition.notify_all();
        }
    });
    fltkWindowThread.detach();
    #endif
    SPDLOG_DEBUG("启动完成");
    {
        std::unique_lock<std::mutex> lock(mutex);
        while(count > 0) {
            condition.wait(lock);
        }
    }
}

int main(const int argc, const char * const argv[]) {
    std::set_terminate([]() {
        std::exit(-9999);
    });
    lifuren::logger::init();
    SPDLOG_DEBUG("启动系统");
    launch();
    SPDLOG_DEBUG("系统退出");
    lifuren::logger::shutdown();
    return 0;
}

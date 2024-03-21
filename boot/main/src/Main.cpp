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

#include <thread>

int main(const int argc, const char * const argv[]) {
    lifuren::logger::init();
    SPDLOG_DEBUG("启动系统");
    #if __REST__
    std::thread httpServerThread(lifuren::initHttpServer);
    #endif
    #if __FLTK__
    std::thread fltkWindowThread(lifuren::initFltkWindow);
    #endif
    SPDLOG_DEBUG("启动完成");
    #if __REST__
    httpServerThread.join();
    #endif
    #if __FLTK__
    fltkWindowThread.join();
    #endif
    SPDLOG_DEBUG("系统退出");
    lifuren::logger::shutdown();
    return 0;
}

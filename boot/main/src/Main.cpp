/**
 * 李夫人 - 玉簪花神
 * 
 * @author acgist
 */
#include "REST.hpp"
#include "Logger.hpp"
#include "Window.hpp"

#include <thread>

int main(const int argc, const char * const argv[]) {
    lifuren::logger::init();
    SPDLOG_DEBUG("启动系统");
    std::thread httpServerThread(lifuren::initHttpServer);
    std::thread fltkThread(lifuren::initWindow);
    SPDLOG_DEBUG("启动完成");
    httpServerThread.join();
    fltkThread.join();
    SPDLOG_DEBUG("系统退出");
    lifuren::logger::shutdown();
    return 0;
}

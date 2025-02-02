#include "lifuren/CLI.hpp"
#if LFR_ENABLE_FLTK
#include "lifuren/FLTK.hpp"
#endif
#if LFR_ENABLE_REST
#include "lifuren/REST.hpp"
#endif

#include <thread>

#include "spdlog/spdlog.h"

#include "lifuren/Config.hpp"
#include "lifuren/Logger.hpp"

static void launch();

int main(const int argc, const char* const argv[]) {
    std::set_terminate([]() {
        std::exit(-9999);
    });
    #if _WIN32
    system("chcp 65001");
    #endif
    lifuren::logger::init();
    lifuren::logger::opencv::init();
    lifuren::config::init(argc, argv);
    SPDLOG_DEBUG("启动系统");
    if(lifuren::cli(argc, argv)) {
        // 执行命令行时不会启动FLTK和REST服务
    } else {
        launch();
    }
    SPDLOG_DEBUG("系统退出");
    lifuren::logger::shutdown();
    return 0;
}

inline static void launch() {
    #if LFR_ENABLE_FLTK
    std::thread fltkThread([]() {
        lifuren::initFltkService();
    });
    #endif
    #if LFR_ENABLE_REST
    std::thread restThread([]() {
        lifuren::initRestService();
    });
    #endif
    SPDLOG_DEBUG("启动完成");
    #if LFR_ENABLE_FLTK
    fltkThread.join();
    #endif
    #if LFR_ENABLE_REST
    restThread.join();
    #endif
}

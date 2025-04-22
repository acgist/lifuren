#include "lifuren/CLI.hpp"

#if LFR_ENABLE_GUI
#include "lifuren/GUI.hpp"
#if _WIN32
#include "WinUser.h"
#endif
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
    #if LFR_ENABLE_GUI
    SetProcessDPIAware();
    #endif
    #endif
    lifuren::logger::init();
    lifuren::logger::opencv::init();
    lifuren::config::init(argc, argv);
    SPDLOG_DEBUG("启动系统");
    if(lifuren::cli(argc, argv)) {
        // 命令行
    } else {
        launch();
    }
    SPDLOG_DEBUG("系统退出");
    lifuren::logger::stop();
    return 0;
}

inline static void launch() {
    #if LFR_ENABLE_GUI
    std::thread guiThread([]() {
        lifuren::initGUI();
    });
    #endif
    SPDLOG_DEBUG("启动完成");
    #if LFR_ENABLE_GUI
    guiThread.join();
    #endif
}

#include "lifuren/CLI.hpp"

#include "spdlog/spdlog.h"

#include "lifuren/Config.hpp"
#include "lifuren/Logger.hpp"

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
    lifuren::cli(argc, argv);
    SPDLOG_DEBUG("系统退出");
    lifuren::logger::stop();
    return 0;
}

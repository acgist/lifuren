/**
 * 李夫人 - 玉簪花神
 * 
 * @author acgist
 */
#include "Logger.hpp"
#include "Window.hpp"

int main(const int argc, const char * const argv[]) {
    lifuren::logger::init();
    SPDLOG_DEBUG("启动系统");
    lifuren::MainWindow* mainPtr = new lifuren::MainWindow(1200, 800, "李夫人");
    mainPtr->init();
    mainPtr->show();
    SPDLOG_DEBUG("启动完成");
    const int code = Fl::run();
    // 释放窗口
    if(mainPtr != nullptr) {
        delete mainPtr;
        mainPtr = nullptr;
    }
    SPDLOG_DEBUG("系统退出");
    lifuren::logger::shutdown();
    return code;
}

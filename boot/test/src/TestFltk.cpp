#include "../src/header/Fltk.hpp"

int main(const int argc, const char * const argv[]) {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    lifuren::LifurenWindow* windowPtr = new lifuren::LifurenWindow(200, 100, "李夫人");
    windowPtr->init();
    windowPtr->show();
    const int code = Fl::run();
    if(windowPtr != nullptr) {
        delete windowPtr;
        windowPtr = nullptr;
    }
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return code;
}

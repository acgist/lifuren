#include "../src/header/Fltk.hpp"

int main(const int argc, const char * const argv[]) {
    lifuren::init(argc, argv);
    LOG(INFO) << "测试";
    lifuren::LifurenWindow* windowPtr = new lifuren::LifurenWindow(200, 100, "李夫人");
    windowPtr->init();
    windowPtr->show();
    const int code = Fl::run();
    LOG(INFO) << "完成";
    lifuren::shutdown();
    delete windowPtr;
    return code;
}

#include "../src/header/Fltk.hpp"

int main(int argc, char const *argv[]) {
    lifuren::init(argc, argv);
    LOG(INFO) << "测试FLTK";
    lifuren::LifurenWindow* windowPtr = new lifuren::LifurenWindow(200, 100, "李夫人");
    windowPtr->init();
    windowPtr->show();
    const int ret = Fl::run();
    lifuren::shutdown();
    return ret;
}

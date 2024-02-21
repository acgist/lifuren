/**
 * 李夫人 - 玉簪花神
 * 
 * @author acgist
 */
#include "GLog.hpp"
#include "Window.hpp"

int main(const int argc, const char * const argv[]) {
    lifuren::init(argc, argv);
    LOG(INFO) << "启动系统";
    lifuren::MainWindow* mainPtr = new lifuren::MainWindow(1200, 800, "李夫人");
    mainPtr->init();
    mainPtr->show();
    const int code = Fl::run();
    LOG(INFO) << "启动完成";
    // 释放窗口
    if(mainPtr != nullptr) {
        delete mainPtr;
        mainPtr = nullptr;
    }
    lifuren::shutdown();
    return code;
}

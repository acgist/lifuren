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
    lifuren::MainWindow* window = new lifuren::MainWindow(512, 256, "李夫人");
    window->show();
    const int code = Fl::run();
    LOG(INFO) << "启动完成";
    lifuren::shutdown();
    // 释放窗口
    if(window != nullptr) {
        delete window;
        window = nullptr;
    }
    return code;
}

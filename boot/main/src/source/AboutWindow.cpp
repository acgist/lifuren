#include "../header/Window.hpp"

lifuren::AboutWindow::AboutWindow(int width, int height, const char* titlePtr) : LFRWindow(width, height, titlePtr) {
}

lifuren::AboutWindow::~AboutWindow() {
    LOG(INFO) << "关闭AboutWindow";
}

void lifuren::AboutWindow::drawElement() {

    

}

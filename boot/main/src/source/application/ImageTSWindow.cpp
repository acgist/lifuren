#include "../../header/Window.hpp"

lifuren::ImageTSWindow::ImageTSWindow(int width, int height, const char* title) :ModelTSWindow(width, height, title) {

}

lifuren::ImageTSWindow::~ImageTSWindow() {
    SPDLOG_DEBUG("关闭窗口：{}", __FILE__);
}

void lifuren::ImageTSWindow::drawElement() {
}

#include "../../header/FLTK.hpp"

lifuren::VideoTSWindow::VideoTSWindow(int width, int height, const char* title) : ModelTSWindow(width, height, title) {
}

lifuren::VideoTSWindow::~VideoTSWindow() {
    SPDLOG_DEBUG("关闭窗口：{}", __FILE__);
}

void lifuren::VideoTSWindow::drawElement() {
}

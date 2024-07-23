#include "lifuren/FLTK.hpp"

#include "spdlog/spdlog.h"

lifuren::VideoGCWindow::VideoGCWindow(int width, int height, const char* title) : ModelGCWindow(width, height, title) {
}

lifuren::VideoGCWindow::~VideoGCWindow() {
    SPDLOG_DEBUG("关闭窗口：{}", __FILE__);
}

void lifuren::VideoGCWindow::drawElement() {
}

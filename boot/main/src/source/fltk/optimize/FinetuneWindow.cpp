#include "lifuren/FLTK.hpp"

#include "spdlog/spdlog.h"

lifuren::FinetuneWindow::FinetuneWindow(int width, int height, const char* title) : Window(width, height, title) {
}

lifuren::FinetuneWindow::~FinetuneWindow() {
    SPDLOG_DEBUG("关闭窗口：{}", __FILE__);
}

void lifuren::FinetuneWindow::drawElement() {
}

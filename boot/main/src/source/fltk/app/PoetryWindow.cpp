#include "lifuren/FLTK.hpp"

#include "spdlog/spdlog.h"

lifuren::PoetryWindow::PoetryWindow(int width, int height, const char* title) :ModelWindow(width, height, title) {
}

lifuren::PoetryWindow::~PoetryWindow() {
    SPDLOG_DEBUG("关闭窗口：{}", __FILE__);
    lifuren::config::saveFile();
}

void lifuren::PoetryWindow::drawElement() {
    // 绘制界面
    // 绑定事件
}

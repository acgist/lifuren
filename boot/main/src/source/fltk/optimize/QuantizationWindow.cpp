#include "lifuren/FLTK.hpp"

#include "spdlog/spdlog.h"

lifuren::QuantizationWindow::QuantizationWindow(int width, int height, const char* title) : Window(width, height, title) {
}

lifuren::QuantizationWindow::~QuantizationWindow() {
    SPDLOG_DEBUG("关闭窗口：{}", __FILE__);
}

void lifuren::QuantizationWindow::drawElement() {
}

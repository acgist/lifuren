#include "lifuren/FLTK.hpp"

#include "spdlog/spdlog.h"

lifuren::DocsMarkWindow::DocsMarkWindow(int width, int height, const char* title) : MarkWindow(width, height, title) {
}

lifuren::DocsMarkWindow::~DocsMarkWindow() {
    SPDLOG_DEBUG("关闭窗口：{}", __FILE__);
}

void lifuren::DocsMarkWindow::drawElement() {
}

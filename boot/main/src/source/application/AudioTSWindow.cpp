#include "../../header/Window.hpp"

lifuren::AudioTSWindow::AudioTSWindow(int width, int height, const char* title) : ModelTSWindow(width, height, title) {
}

lifuren::AudioTSWindow::~AudioTSWindow() {
    SPDLOG_DEBUG("关闭窗口：{}", __FILE__);
}

void lifuren::AudioTSWindow::drawElement() {
}

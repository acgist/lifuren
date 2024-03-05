#include "../../header/Window.hpp"

lifuren::PoetryGCWindow::PoetryGCWindow(int width, int height, const char* title) : ModelGCWindow(width, height, title) {
}

lifuren::PoetryGCWindow::~PoetryGCWindow() {
    LFR_DELETE_THIS_PTR(autoMarkPtr);
}

void lifuren::PoetryGCWindow::drawElement() {
}

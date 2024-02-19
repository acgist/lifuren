#include "../header/Window.hpp"

lifuren::MainWindow::MainWindow(int width, int height, const char* titlePtr) : LFRWindow(width, height, titlePtr) {
}

lifuren::MainWindow::~MainWindow() {
}

void lifuren::MainWindow::init() {
    lifuren::LFRWindow::init();
}

void lifuren::MainWindow::about() {
}

void lifuren::MainWindow::setting() {
}

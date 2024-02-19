#include "../header/Window.hpp"

lifuren::LFRWindow::LFRWindow(int width, int height, const char* titlePtr) : Fl_Window(width, height, titlePtr) {
    this->begin();
    this->init();
    this->end();
}

lifuren::LFRWindow::~LFRWindow() {
}

void lifuren::LFRWindow::init() {
    this->icon();
    this->center();
}

void lifuren::LFRWindow::icon() {
    Fl_PNG_Image iconImage("images/logo.png");
    Fl_Window::default_icon(&iconImage);
}

void lifuren::LFRWindow::center() {
    this->position(200, 200);
}
#include "../header/Window.hpp"

lifuren::LFRWindow::LFRWindow(int width, int height, const char* titlePtr) : Fl_Window(width, height, titlePtr) {
    this->begin();
    this->init();
    this->end();
}

lifuren::LFRWindow::~LFRWindow() {
    if(this->iconImagePtr != nullptr) {
        delete this->iconImagePtr;
        this->iconImagePtr = nullptr;
    }
}

void lifuren::LFRWindow::init() {
    this->icon();
    this->center();
}

void lifuren::LFRWindow::icon() {
    Fl_PNG_Image iconImage("../images/logo.png");
    this->iconImagePtr = static_cast<Fl_PNG_Image*>(iconImage.copy(32, 32));
    Fl_Window::default_icon(this->iconImagePtr);
}

void lifuren::LFRWindow::center() {
    this->position(200, 200);
}
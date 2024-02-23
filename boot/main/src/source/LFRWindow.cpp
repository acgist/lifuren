#include "../header/Window.hpp"

/**
 * @param source 原始值
 * @param target 比较值
 * 
 * @return 绝对值
 */
int abs(int source, int target);

lifuren::LFRWindow::LFRWindow(int width, int height, const char* titlePtr) : Fl_Window(width, height, titlePtr) {
}

lifuren::LFRWindow::~LFRWindow() {
    if(this->iconImagePtr != nullptr) {
        delete this->iconImagePtr;
        this->iconImagePtr = nullptr;
    }
}

void lifuren::LFRWindow::init() {
    this->begin();
    this->icon();
    this->center();
    this->drawElement();
    this->end();
}

void lifuren::LFRWindow::icon() {
    const char* iconPath = "../images/logo.png";
    SPDLOG_DEBUG("加载图标：{}", iconPath);
    Fl_PNG_Image iconImage(iconPath);
    this->iconImagePtr = static_cast<Fl_PNG_Image*>(iconImage.copy(32, 32));
    Fl_Window::default_icon(this->iconImagePtr);
}

void lifuren::LFRWindow::center() {
    const int fullWidth  = Fl::w();
    const int fullHeight = Fl::h();
    const int width  = this->w();
    const int height = this->h();
    this->position(abs(fullWidth, width) / 2, abs(fullHeight, height) / 2);
}

int abs(int source, int target) {
    if(source > target) {
        return source - target;
    } else {
        return target - source;
    }
}
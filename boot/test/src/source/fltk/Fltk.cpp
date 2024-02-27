#include "../../header/Fltk.hpp"

/**
 * 点击事件
 */
void buttonCallback(Fl_Widget* widgetPtr, void* voidPtr) {
    SPDLOG_DEBUG("你点击了按钮（local）：{}", ((lifuren::LifurenWindow*) voidPtr)->inputPtr->value());
}

/**
 * 点击事件代理
 */
void buttonCallbackProxy(Fl_Widget* widgetPtr, void* voidPtr) {
    ((lifuren::LifurenWindow*) voidPtr)->buttonCallback(widgetPtr, voidPtr);
}

lifuren::LifurenWindow::LifurenWindow(int width, int height, const char* title) : Fl_Window(width, height, title) {
}

lifuren::LifurenWindow::~LifurenWindow() {
    if(this->inputPtr != nullptr) {
        delete this->inputPtr;
        this->inputPtr = nullptr;
    }
    if(this->buttonPtr != nullptr) {
        delete this->buttonPtr;
        this->buttonPtr = nullptr;
    }
    if(this->buttonProxyPtr != nullptr) {
        delete this->buttonProxyPtr;
        this->buttonProxyPtr = nullptr;
    }
}

void lifuren::LifurenWindow::init() {
    this->begin();
    this->resizable(this);
    this->inputPtr  = new Fl_Input(50,  10, 100, 20, "年纪");
    this->buttonPtr = new Fl_Button(30, 40, 60,  20, "提交");
    this->buttonPtr->callback(::buttonCallback, this);
    this->buttonProxyPtr = new Fl_Button(110, 40, 60, 20, "提交");
    this->buttonProxyPtr->callback(::buttonCallbackProxy, this);
    this->end();
}

/**
 * 如果想要按钮直接调用需要改为静态函数
 */
void lifuren::LifurenWindow::buttonCallback(Fl_Widget* widgetPtr, void* voidPtr) {
    SPDLOG_DEBUG("你点击了按钮（this）：{}", this->inputPtr->value());
}

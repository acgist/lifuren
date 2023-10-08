#include "../../header/Fltk.hpp"

void buttonCallbackEx(Fl_Widget* widgetPtr,  void* voidPtr) {
    LOG(INFO) << "你点击了按钮";
}

lifuren::LifurenWindow::LifurenWindow(int width, int height, const char* titlePtr) : Fl_Window(width, height, titlePtr) {
}

void lifuren::LifurenWindow::buttonCallback(Fl_Widget* widgetPtr, void* voidPtr) {
    LOG(INFO) << "你点击了按钮";
}

void lifuren::LifurenWindow::init() {
    this->begin();
    this->resizable(this);
    Fl_Input*  inputPtr  = new Fl_Input(10,  10, 100, 20, "年纪");
    Fl_Button* buttonPtr = new Fl_Button(10, 30, 60, 20, "提交");
    buttonPtr->callback(buttonCallback, this);
    // buttonPtr->callback(buttonCallbackEx, this);
    // buttonPtr->callback(::buttonCallbackEx, this);
    this->end();
}

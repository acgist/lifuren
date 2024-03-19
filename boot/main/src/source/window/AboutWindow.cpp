#include "../../header/Window.hpp"

#include "FL/filename.H"

lifuren::AboutWindow::AboutWindow(int width, int height, const char* title) : LFRWindow(width, height, title) {
}

lifuren::AboutWindow::~AboutWindow() {
    SPDLOG_DEBUG("关闭窗口：{}", __FILE__);
    LFR_DELETE_THIS_PTR(homePagePtr);
    // 注意顺序：display->buffer
    LFR_DELETE_THIS_PTR(aboutDisplayPtr);
    LFR_DELETE_THIS_PTR(aboutBufferPtr);
}

void lifuren::AboutWindow::drawElement() {
    this->aboutDisplayPtr = new Fl_Text_Display(10, 30, this->w() - 20, this->h() - 90, "关于");
    this->aboutDisplayPtr->wrap_mode(this->aboutDisplayPtr->WRAP_AT_COLUMN, this->aboutDisplayPtr->textfont());
    this->aboutDisplayPtr->color(FL_BACKGROUND_COLOR);
    // 内容
    this->aboutBufferPtr = new Fl_Text_Buffer();
    this->aboutBufferPtr->text("李夫人，这是一个研究生成网络、机器视觉、自然语言处理的程序。\n");
    this->aboutBufferPtr->append("Gitee:https://gitee.com/acgist/lifuren\n");
    this->aboutBufferPtr->append("Github：https://github.com/acgist/lifuren\n");
    this->aboutBufferPtr->append("作者：acgist");
    this->aboutDisplayPtr->buffer(this->aboutBufferPtr);
    // 主页
    this->homePagePtr = new Fl_Button(this->w() / 2 - 40, this->h() - 40, 80, 30, "主页");
    this->homePagePtr->callback([](Fl_Widget* widgetPtr, void* voidPtr) -> void {
        const int ret = fl_open_uri("https://gitee.com/acgist/lifuren");
        SPDLOG_DEBUG("打开主页：{}", ret);
    }, this);
}

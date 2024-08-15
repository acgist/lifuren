#include "lifuren/FLTK.hpp"

#include "spdlog/spdlog.h"

#include "FL/filename.H"
#include "FL/Fl_Button.H"
#include "FL/Fl_Text_Buffer.H"
#include "FL/Fl_Text_Display.H"

static Fl_Button*       homePagePtr    { nullptr };
static Fl_Text_Buffer*  aboutBufferPtr { nullptr };
static Fl_Text_Display* aboutDisplayPtr{ nullptr };

lifuren::AboutWindow::AboutWindow(int width, int height, const char* title) : Window(width, height, title) {
}

lifuren::AboutWindow::~AboutWindow() {
    SPDLOG_DEBUG("关闭窗口：{}", __FILE__);
    LFR_DELETE_PTR(homePagePtr);
    LFR_DELETE_PTR(aboutDisplayPtr);
    LFR_DELETE_PTR(aboutBufferPtr);
}

void lifuren::AboutWindow::drawElement() {
    // 内容
    aboutDisplayPtr = new Fl_Text_Display(10, 30, this->w() - 20, this->h() - 90, "关于");
    aboutBufferPtr  = new Fl_Text_Buffer();
    aboutDisplayPtr->color(FL_BACKGROUND_COLOR);
    aboutDisplayPtr->buffer(aboutBufferPtr);
    aboutDisplayPtr->wrap_mode(aboutDisplayPtr->WRAP_AT_COLUMN, aboutDisplayPtr->textfont());
    aboutBufferPtr->text("李夫人，这是一个研究生成网络、机器视觉、自然语言处理的程序。\n");
    aboutBufferPtr->append("Gitee：https://gitee.com/acgist/lifuren\n");
    aboutBufferPtr->append("Github：https://github.com/acgist/lifuren\n");
    aboutBufferPtr->append("作者：碧螺萧萧（acgist）");
    aboutDisplayPtr->end();
    // 主页
    homePagePtr = new Fl_Button(this->w() / 2 - 40, this->h() - 40, 80, 30, "主页");
    homePagePtr->callback([](Fl_Widget* widgetPtr, void* voidPtr) -> void {
        const int ret = fl_open_uri("https://gitee.com/acgist/lifuren");
        SPDLOG_DEBUG("打开主页：{}", ret);
    }, this);
}

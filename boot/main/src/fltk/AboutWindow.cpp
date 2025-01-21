#include "lifuren/FLTK.hpp"

#include "spdlog/spdlog.h"

#include "FL/filename.H"
#include "FL/Fl_Button.H"
#include "FL/Fl_Text_Buffer.H"
#include "FL/Fl_Text_Display.H"

#include "lifuren/Raii.hpp"

static Fl_Button      * homePagePtr    { nullptr };
static Fl_Text_Buffer * aboutBufferPtr { nullptr };
static Fl_Text_Display* aboutDisplayPtr{ nullptr };

lifuren::AboutWindow::AboutWindow(int width, int height, const char* title) : Window(width, height, title) {
}

lifuren::AboutWindow::~AboutWindow() {
    LFR_DELETE_PTR(homePagePtr);
    LFR_DELETE_PTR(aboutDisplayPtr);
    LFR_DELETE_PTR(aboutBufferPtr);
}

void lifuren::AboutWindow::drawElement() {
    aboutDisplayPtr = new Fl_Text_Display(10, 30, this->w() - 20, this->h() - 90, "关于");
    aboutBufferPtr  = new Fl_Text_Buffer();
    aboutDisplayPtr->begin();
    aboutDisplayPtr->color(FL_BACKGROUND_COLOR);
    aboutDisplayPtr->buffer(aboutBufferPtr);
    aboutDisplayPtr->wrap_mode(aboutDisplayPtr->WRAP_AT_COLUMN, aboutDisplayPtr->textfont());
    aboutDisplayPtr->end();
    homePagePtr = new Fl_Button(this->w() / 2 - 40, this->h() - 40, 80, 30, "主页");
}

void lifuren::AboutWindow::bindEvent() {
    homePagePtr->callback([](Fl_Widget* widgetPtr, void* voidPtr) -> void {
        const int ret = fl_open_uri("https://gitee.com/acgist/lifuren");
        SPDLOG_DEBUG("打开主页：{}", ret);
    }, this);
}

void lifuren::AboutWindow::fillData() {
    aboutBufferPtr->text(R"(李夫人
    
北方有佳人，绝世而独立。
一顾倾人城，再顾倾人国。
宁不知倾城与倾国，佳人难再得。

项目主要研发方向是生成网络、机器视觉、自然语言处理。    

https://gitee.com/acgist/lifuren
https://github.com/acgist/lifuren

Copyright(c) 2024-present acgist. All Rights Reserved.

http://www.apache.org/licenses/LICENSE-2.0
)");
}

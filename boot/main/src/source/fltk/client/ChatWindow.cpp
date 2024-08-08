#include "lifuren/FLTK.hpp"

#include <algorithm>

#include "spdlog/spdlog.h"

#include "FL/Fl_Button.H"

static Fl_Button* sendPtr  { nullptr };
static Fl_Button* stopPtr  { nullptr };
static Fl_Button* configPtr{ nullptr };

// 是否停止
static bool stop{ true };
// 配置窗口
static lifuren::ChatConfigWindow* chatConfigWindowPtr{ nullptr };

lifuren::ChatWindow::ChatWindow(int width, int height, const char* title) : ModelWindow(width, height, title) {
}

lifuren::ChatWindow::~ChatWindow() {
    SPDLOG_DEBUG("关闭窗口：{}", __FILE__);
    stop = true;
    LFR_DELETE_PTR(sendPtr);
    LFR_DELETE_PTR(stopPtr);
    LFR_DELETE_PTR(configPtr);
    LFR_DELETE_PTR(chatConfigWindowPtr);
    LFR_DELETE_THIS_PTR(clientPtr);
}

void lifuren::ChatWindow::drawElement() {
    sendPtr   = new Fl_Button(this->w() - 120, this->h() - 50, 100, 30, "发送消息");
    stopPtr   = new Fl_Button(this->w() - 230, this->h() - 50, 100, 30, "结束回答");
    configPtr = new Fl_Button(this->w() - 340, this->h() - 50, 100, 30, "⚙配置");
    // 发送
    sendPtr->callback([](Fl_Widget*, void*) {
    });
    // 结束
    stopPtr->callback([](Fl_Widget*, void*) {
        stop = true;
    });
    // 配置
    configPtr->callback([](Fl_Widget*, void* voidPtr) {
        chatConfigWindowPtr = new lifuren::ChatConfigWindow{620, 800};
        chatConfigWindowPtr->init();
        chatConfigWindowPtr->show();
        chatConfigWindowPtr->callback([](Fl_Widget*, void*) {
            chatConfigWindowPtr->hide();
            LFR_DELETE_PTR(chatConfigWindowPtr);
        }, voidPtr);
    }, this);
}

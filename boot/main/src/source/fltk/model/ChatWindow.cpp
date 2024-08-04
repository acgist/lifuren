#include "lifuren/FLTK.hpp"

#include "spdlog/spdlog.h"

#include <algorithm>

lifuren::ChatWindow::ChatWindow(int width, int height, const char* title) : ModelWindow(width, height, title) {
    this->chatConfigPtr = &lifuren::config::CONFIG.chat;
}

lifuren::ChatWindow::~ChatWindow() {
    SPDLOG_DEBUG("关闭窗口：{}", __FILE__);
    lifuren::config::saveFile();
}

void lifuren::ChatWindow::drawElement() {
    this->modelPtr = new Fl_Choice(100, 10, this->w() - 200, 30, "模型名称");
    std::for_each(lifuren::config::nlpClients.begin(), lifuren::config::nlpClients.end(), [this](const auto& v) {
        this->modelPtr->add(v.c_str());
    });
    auto defaultPtr = this->modelPtr->find_item(this->chatConfigPtr->model.c_str());
    if(defaultPtr) {
        this->modelPtr->value(defaultPtr); 
    }
    this->modelPtr->callback([](Fl_Widget*, void* voidPtr) {
        ChatWindow* windowPtr = static_cast<ChatWindow*>(voidPtr);
        windowPtr->chatConfigPtr->model = windowPtr->modelPtr->text();
    }, this);
    this->trainStartPtr = new Fl_Button(10,  50, 100, 30, "开始训练");
    this->trainStopPtr  = new Fl_Button(120, 50, 100, 30, "结束训练");
}

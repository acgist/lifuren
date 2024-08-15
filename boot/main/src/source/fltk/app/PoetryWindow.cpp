#include "lifuren/FLTK.hpp"

#include "spdlog/spdlog.h"

lifuren::PoetryWindow::PoetryWindow(int width, int height, const char* title) :ModelWindow(width, height, title) {
}

lifuren::PoetryWindow::~PoetryWindow() {
    SPDLOG_DEBUG("关闭窗口：{}", __FILE__);
    lifuren::config::saveFile();
}

void lifuren::PoetryWindow::drawElement() {
    // this->modelPtr = new Fl_Input_Directory_Chooser(110, 10, this->w() - 200, 30, "模型名称");
    // this->modelPtr->value(this->videoConfigPtr->model.c_str());
    // LFR_INPUT_DIRECTORY_CHOOSER(modelPtr, videoConfigPtr, model, PoetryWindow);
    // this->trainStartPtr = new Fl_Button(10,  50, 100, 30, "开始训练");
    // this->trainStopPtr  = new Fl_Button(120, 50, 100, 30, "结束训练");
}

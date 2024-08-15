#include "lifuren/FLTK.hpp"

#include "spdlog/spdlog.h"

// 开始训练
static void trainStart(Fl_Widget*, void*);
// 结束训练
static void trainStop(Fl_Widget*, void*);

lifuren::ImageWindow::ImageWindow(int width, int height, const char* title) : ModelWindow(width, height, title) {
}

lifuren::ImageWindow::~ImageWindow() {
    SPDLOG_DEBUG("关闭窗口：{}", __FILE__);
    lifuren::config::saveFile();
}

void lifuren::ImageWindow::drawElement() {
    // this->modelPtr = new Fl_Input_Directory_Chooser(110, 10, this->w() - 200, 30, "模型名称");
    // this->modelPtr->value(this->imageConfigPtr->model.c_str());
    // LFR_INPUT_DIRECTORY_CHOOSER(modelPtr, imageConfigPtr, model, ImageWindow);
    // this->trainStartPtr = new Fl_Button(10,  50, 100, 30, "开始训练");
    // this->trainStopPtr  = new Fl_Button(120, 50, 100, 30, "结束训练");
    // this->trainStartPtr->callback(trainStart, this);
    // this->trainStopPtr->callback(trainStop, this);
}

static void trainStart(Fl_Widget* widgetPtr, void* voidPtr) {
    const lifuren::ImageWindow* windowPtr = (lifuren::ImageWindow*) voidPtr;
}

static void trainStop(Fl_Widget* widgetPtr, void* voidPtr) {
    const lifuren::ImageWindow* windowPtr = (lifuren::ImageWindow*) voidPtr;
}

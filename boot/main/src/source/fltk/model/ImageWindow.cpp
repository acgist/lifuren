#include "lifuren/FLTK.hpp"

#include "spdlog/spdlog.h"

// 开始训练
static void trainStart(Fl_Widget*, void*);
// 结束训练
static void trainStop(Fl_Widget*, void*);

lifuren::ImageWindow::ImageWindow(int width, int height, const char* title) : ModelWindow(width, height, title) {
    this->loadConfig("ImageGC");
}

lifuren::ImageWindow::~ImageWindow() {
    SPDLOG_DEBUG("关闭窗口：{}", __FILE__);
    lifuren::config::saveFile(CONFIGS_PATH);
}

void lifuren::ImageWindow::drawElement() {
    this->modelPathPtr = new Fl_Input_Directory_Chooser(100, 10, this->w() - 200, 30, "模型目录");
    this->modelPathPtr->value(this->configPtr->modelPath.c_str());
    this->datasetPathPtr = new Fl_Input_Directory_Chooser(100, 50, this->w() - 200, 30, "数据目录");
    this->datasetPathPtr->value(this->configPtr->datasetPath.c_str());
    LFR_INPUT_DIRECTORY_CHOOSER(modelPathPtr, modelPath, ImageWindow);
    this->trainStartPtr = new Fl_Button(10,  90, 100, 30, "开始训练");
    this->trainStopPtr  = new Fl_Button(120, 90, 100, 30, "结束训练");
    this->trainStartPtr->callback(trainStart, this);
    this->trainStopPtr->callback(trainStop, this);
}

static void trainStart(Fl_Widget* widgetPtr, void* voidPtr) {
    const lifuren::ImageWindow* windowPtr = (lifuren::ImageWindow*) voidPtr;
}

static void trainStop(Fl_Widget* widgetPtr, void* voidPtr) {
    const lifuren::ImageWindow* windowPtr = (lifuren::ImageWindow*) voidPtr;
}

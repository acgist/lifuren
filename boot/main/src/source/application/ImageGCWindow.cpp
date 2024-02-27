#include "../../header/Window.hpp"

#include "Fl/Fl_Input_Choice.H"

lifuren::ImageGCWindow::ImageGCWindow(int width, int height, const char* title) : LFRWindow(width, height, title) {
}

lifuren::ImageGCWindow::~ImageGCWindow() {
    SETTINGS.saveFile(SETTINGS_PATH);
    SPDLOG_DEBUG("关闭ImageGCWindow");
    LFR_DELETE_PTR(modelPathPtr);
    LFR_DELETE_PTR(datasetPathPtr);
}

void lifuren::ImageGCWindow::drawElement() {
    this->modelPathPtr = new Fl_Input_Directory_Chooser(100, 10, this->w() - 200, 30, "模型目录");
    this->datasetPathPtr = new Fl_Input_Directory_Chooser(100, 50, this->w() - 200, 30, "数据目录");
}

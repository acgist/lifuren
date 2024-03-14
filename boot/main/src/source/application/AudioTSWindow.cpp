#include "../../header/Window.hpp"

#include "utils/Jsons.hpp"

lifuren::AudioTSWindow::AudioTSWindow(int width, int height, const char* title) : ModelTSWindow(width, height, title) {
    this->loadSetting("AudioTS");
}

lifuren::AudioTSWindow::~AudioTSWindow() {
    SPDLOG_DEBUG("关闭窗口：{}", __FILE__);
    lifuren::jsons::saveFile(SETTINGS_PATH, lifuren::SETTINGS);
}

void lifuren::AudioTSWindow::drawElement() {
    this->modelPathPtr = new Fl_Input_Directory_Chooser(100, 10, this->w() - 200, 30, "模型目录");
    this->modelPathPtr->value(this->settingPtr->modelPath.c_str());
    this->datasetPathPtr = new Fl_Input_Directory_Chooser(100, 50, this->w() - 200, 30, "数据目录");
    this->datasetPathPtr->value(this->settingPtr->datasetPath.c_str());
    // LFR_INPUT_DIRECTORY_CHOOSER_CALLBACK(modelPathPtr, modelPath, ImageGCWindow);
    // LFR_INPUT_DIRECTORY_CHOOSER_CALLBACK_CALL(datasetPathPtr, datasetPath, ImageGCWindow, loadImageVector);
    this->trainStartPtr = new Fl_Button(10,  90, 100, 30, "开始训练");
    this->trainStopPtr  = new Fl_Button(120, 90, 100, 30, "结束训练");
    this->transferPtr   = new Fl_Button(230, 90, 100, 30, "风格迁移");
}

#include "../../header/Window.hpp"

lifuren::ImageGCWindow::ImageGCWindow(int width, int height, const char* title) : ModelGCWindow(width, height, title) {
    auto iterator = SETTINGS.settings.find("ImageGC");
    if(iterator == SETTINGS.settings.end()) {
        this->settingPtr = new Setting();
        SETTINGS.settings.insert(std::make_pair("ImageGC", *this->settingPtr));
    } else {
        this->settingPtr = &iterator->second;
    }
}

lifuren::ImageGCWindow::~ImageGCWindow() {
    SETTINGS.saveFile(SETTINGS_PATH);
    SPDLOG_DEBUG("关闭ImageGCWindow");
    LFR_DELETE_PTR(modelPathPtr);
    LFR_DELETE_PTR(datasetPathPtr);
    LFR_DELETE_PTR(prevPtr);
    LFR_DELETE_PTR(nextPtr);
    LFR_DELETE_PTR(trainStartPtr);
    LFR_DELETE_PTR(trainStopPtr);
    LFR_DELETE_PTR(generatePtr);
}

void lifuren::ImageGCWindow::drawElement() {
    this->modelPathPtr = new Fl_Input_Directory_Chooser(100, 10, this->w() - 200, 30, "模型目录");
    this->modelPathPtr->value(this->settingPtr->modelPath.c_str());
    this->datasetPathPtr = new Fl_Input_Directory_Chooser(100, 50, this->w() - 200, 30, "数据目录");
    this->datasetPathPtr->value(this->settingPtr->datasetPath.c_str());
    LFR_INPUT_DIRECTORY_CHOOSER_CALLBACK(modelPathPtr, modelPath, ImageGCWindow);
    LFR_INPUT_DIRECTORY_CHOOSER_CALLBACK(datasetPathPtr, datasetPath, ImageGCWindow);
    this->prevPtr = new Fl_Button(10,  90, 100, 30, "上一张图");
    this->nextPtr = new Fl_Button(120, 90, 100, 30, "下一张图");
    this->trainStartPtr = new Fl_Button(230, 90, 100, 30, "开始训练");
    this->trainStopPtr  = new Fl_Button(340, 90, 100, 30, "结束训练");
    this->generatePtr   = new Fl_Button(450, 90, 100, 30, "生成图片");
}

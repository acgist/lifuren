#include "../../header/REST.hpp"
#include "../../header/Window.hpp"

void lifuren::initWindow() {
    SPDLOG_INFO("启动FLTK服务");
    lifuren::MainWindow* mainPtr = new lifuren::MainWindow(1200, 800, "李夫人");
    mainPtr->init();
    mainPtr->show();
    const int code = Fl::run();
    // 释放窗口
    if(mainPtr != nullptr) {
        delete mainPtr;
        mainPtr = nullptr;
    }
    SPDLOG_INFO("结束FLTK服务：{}", code);
    lifuren::shutdownHttpServer();
}

std::string lifuren::fileChooser(const char* title, const char* directory, const char* filter) {
    Fl_Native_File_Chooser chooser(Fl_Native_File_Chooser::BROWSE_FILE);
    chooser.title(title);
    chooser.filter(filter);
    chooser.directory(directory);
    const int code = chooser.show();
    switch(code) {
        case 0: {
            const char* filename = chooser.filename();
            SPDLOG_DEBUG("文件选择成功：{} - {}", title, filename);
            return filename;
        }
        default:
            SPDLOG_DEBUG("文件选择失败：{} - {}", title, code);
            return "";
    }
}

std::string lifuren::directoryChooser(const char* title, const char* directory) {
    Fl_Native_File_Chooser chooser(Fl_Native_File_Chooser::BROWSE_DIRECTORY);
    chooser.title(title);
    chooser.directory(directory);
    const int code = chooser.show();
    switch(code) {
        case 0: {
            const char* filename = chooser.filename();
            SPDLOG_DEBUG("目录选择成功：{} - {}", title, filename);
            return filename;
        }
        default:
            SPDLOG_DEBUG("目录选择失败：{} - {}", title, code);
            return "";
    }
}

lifuren::Fl_Input_Directory_Chooser::Fl_Input_Directory_Chooser(
    int x,
    int y,
    int width,
    int height,
    const char* title,
    const char* directory
) :
    Fl_Input(x, y, width, height),
    title(title),
    directory(directory)
{
    this->label(title);
}

lifuren::Fl_Input_Directory_Chooser::~Fl_Input_Directory_Chooser() {
}

int lifuren::Fl_Input_Directory_Chooser::handle(int event) {
    if(event == FL_LEFT_MOUSE) {
        const std::string filename = directoryChooser(this->title);
        if(filename.empty()) {
            return 0;
        }
        this->value(filename.c_str());
        this->do_callback();
        return 0;
    }
    return Fl_Input::handle(event);
}

lifuren::ModelWindow::ModelWindow(int width, int height, const char* title) : LFRWindow(width, height, title) {
}

lifuren::ModelWindow::~ModelWindow() {
    SPDLOG_DEBUG("关闭窗口：{}", __func__);
    LFR_DELETE_THIS_PTR(modelPathPtr);
    LFR_DELETE_THIS_PTR(datasetPathPtr);
}

void lifuren::ModelWindow::loadSetting(const std::string& modelType) {
    auto iterator = SETTINGS.find(modelType);
    if(iterator == SETTINGS.end()) {
        this->settingPtr = new Setting();
        // TODO：BUG拷贝
        SETTINGS.insert(std::make_pair(modelType, *this->settingPtr));
    } else {
        this->settingPtr = &iterator->second;
    }
}

lifuren::ModelGCWindow::ModelGCWindow(int width, int height, const char* title) : ModelWindow(width, height, title) {
}

lifuren::ModelGCWindow::~ModelGCWindow() {
    SPDLOG_DEBUG("关闭窗口：{}", __func__);
    LFR_DELETE_THIS_PTR(prevPtr);
    LFR_DELETE_THIS_PTR(nextPtr);
    LFR_DELETE_THIS_PTR(trainStartPtr);
    LFR_DELETE_THIS_PTR(trainStopPtr);
    LFR_DELETE_THIS_PTR(generatePtr);
}

lifuren::ModelTSWindow::ModelTSWindow(int width, int height, const char* title) : ModelWindow(width, height, title) {
}

lifuren::ModelTSWindow::~ModelTSWindow() {
    SPDLOG_DEBUG("关闭窗口：{}", __func__);
    LFR_DELETE_THIS_PTR(trainStartPtr);
    LFR_DELETE_THIS_PTR(trainStopPtr);
    LFR_DELETE_THIS_PTR(transferPtr);
}

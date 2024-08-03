#if LFR_ENABLE_REST
#include "lifuren/REST.hpp"
#endif
#include "lifuren/FLTK.hpp"

#include "lifuren/Ptr.hpp"

#include "spdlog/spdlog.h"

#include "Fl/Fl_Native_File_Chooser.H"

// 是否关闭
static bool fltkClose = false;

/**
 * @param source 原始值
 * @param target 比较值
 * 
 * @return 绝对值
 */
static int abs(const int& source, const int& target);

void lifuren::initFltkWindow() {
    SPDLOG_INFO("启动FLTK服务");
    lifuren::MainWindow* mainPtr = new lifuren::MainWindow(1200, 800, "李夫人");
    mainPtr->init();
    mainPtr->show();
    const int code = Fl::run();
    LFR_DELETE_PTR(mainPtr);
    SPDLOG_INFO("结束FLTK服务：{}", code);
    #if LFR_ENABLE_REST
    lifuren::shutdownHttpServer();
    #endif
}

void lifuren::shutdownFltkWindow() {
    if(fltkClose) {
        return;
    }
    fltkClose = true;
    while (Fl::first_window()) {
        Fl::first_window()->hide();
    }
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

lifuren::Window::Window(int width, int height, const char* title) : Fl_Window(width, height, title) {
}

lifuren::Window::~Window() {
    if(this->iconImagePtr != nullptr) {
        delete this->iconImagePtr;
        this->iconImagePtr = nullptr;
    }
}

void lifuren::Window::init() {
    this->begin();
    this->icon();
    this->center();
    this->drawElement();
    this->end();
}

void lifuren::Window::icon() {
    const char* iconPath = "../images/logo.png";
    SPDLOG_DEBUG("加载图标：{}", iconPath);
    Fl_PNG_Image iconImage(iconPath);
    this->iconImagePtr = static_cast<Fl_PNG_Image*>(iconImage.copy(32, 32));
    Fl_Window::default_icon(this->iconImagePtr);
}

void lifuren::Window::center() {
    const int fullWidth  = Fl::w();
    const int fullHeight = Fl::h();
    const int width  = this->w();
    const int height = this->h();
    this->position(abs(fullWidth, width) / 2, abs(fullHeight, height) / 2);
}

void lifuren::Window::loadConfig(const std::string& name) {
    SPDLOG_DEBUG("加载窗口配置：{}", name);
    auto iterator = CONFIGS.find(name);
    if(iterator == CONFIGS.end()) {
        auto pair = CONFIGS.emplace(name, Config{});
        this->configPtr = &pair.first->second;
    } else {
        this->configPtr = &iterator->second;
    }
}

lifuren::MarkWindow::MarkWindow(int width, int height, const char* title) : Window(width, height, title) {
}

lifuren::MarkWindow::~MarkWindow() {
    SPDLOG_DEBUG("关闭窗口");
    LFR_DELETE_THIS_PTR(prevPtr);
    LFR_DELETE_THIS_PTR(nextPtr);
    LFR_DELETE_THIS_PTR(datasetPathPtr);
}

lifuren::ModelWindow::ModelWindow(int width, int height, const char* title) : Window(width, height, title) {
}

lifuren::ModelWindow::~ModelWindow() {
    SPDLOG_DEBUG("关闭窗口");
    LFR_DELETE_THIS_PTR(modelPathPtr);
    LFR_DELETE_THIS_PTR(datasetPathPtr);
    LFR_DELETE_THIS_PTR(trainStartPtr);
    LFR_DELETE_THIS_PTR(trainStopPtr);
}

static int abs(const int& source, const int& target) {
    if(source > target) {
        return source - target;
    } else {
        return target - source;
    }
}

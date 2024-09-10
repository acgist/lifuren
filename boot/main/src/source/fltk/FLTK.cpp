#include "lifuren/FLTK.hpp"

#if LFR_ENABLE_REST
#include "lifuren/REST.hpp"
#endif

#include <cmath>
#include <algorithm>

#include "lifuren/Ptr.hpp"

#include "spdlog/spdlog.h"

#include "FL/Fl_Choice.H"
#include "Fl/Fl_Native_File_Chooser.H"

// 是否关闭
static bool fltkClose = false;

void lifuren::initFltkWindow() {
    SPDLOG_INFO("启动FLTK服务");
    lifuren::MainWindow* mainPtr = new lifuren::MainWindow(LFR_WINDOW_WIDTH, LFR_WINDOW_HEIGHT, "李夫人");
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
            std::string filename = chooser.filename();
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
            std::string filename = chooser.filename();
            SPDLOG_DEBUG("目录选择成功：{} - {}", title, filename);
            return filename;
        }
        default:
            SPDLOG_DEBUG("目录选择失败：{} - {}", title, code);
            return "";
    }
}

void lifuren::fillChoice(Fl_Choice* choice, const std::set<std::string>& set, const std::string& value) {
    std::for_each(set.begin(), set.end(), [&value, &choice](const auto& v) {
        const int index = choice->add(v.c_str());
        if(v == value) {
            choice->value(index);
        }
    });
}

lifuren::Window::Window(int width, int height, const char* title) : Fl_Window(width, height, title) {
    SPDLOG_DEBUG("打开窗口");
}

lifuren::Window::~Window() {
    SPDLOG_DEBUG("关闭窗口");
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
    this->position(std::abs(fullWidth - width) / 2, std::abs(fullHeight - height) / 2);
}

void lifuren::Configuration::saveConfig() {
    lifuren::config::saveFile();
}

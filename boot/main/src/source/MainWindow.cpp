#include "../header/Window.hpp"

// 回调绑定
#ifndef CALLBACK_BINDER
#define CALLBACK_BINDER(name)                                                         \
    this->name##ButtonPtr->callback([](Fl_Widget* widgetPtr, void* voidPtr) -> void { \
        ((MainWindow*) voidPtr)->name();                                              \
    }, this);
#endif

// 删除资源指针
#ifndef DELETE_MEDIA_PTR
#define DELETE_MEDIA_PTR(mediaType)               \
    SPDLOG_DEBUG("释放" #mediaType "资源");        \
    if(this->mediaType##GcPtr != nullptr) {       \
        delete this->mediaType##GcPtr;            \
        this->mediaType##GcPtr = nullptr;         \
    }                                             \
    if(this->mediaType##TsPtr != nullptr) {       \
        delete this->mediaType##TsPtr;            \
        this->mediaType##TsPtr = nullptr;         \
    }                                             \
    if(this->mediaType##GcWindowPtr != nullptr) { \
        delete this->mediaType##GcWindowPtr;      \
        this->mediaType##GcWindowPtr = nullptr;   \
    }                                             \
    if(this->mediaType##TsWindowPtr != nullptr) { \
        delete this->mediaType##TsWindowPtr;      \
        this->mediaType##TsWindowPtr = nullptr;   \
    }                                             \
    if(this->mediaType##GroupPtr != nullptr) {    \
        delete this->mediaType##GroupPtr;         \
        this->mediaType##GroupPtr = nullptr;      \
    }
#endif

lifuren::MainWindow::MainWindow(int width, int height, const char* titlePtr) : LFRWindow(width, height, titlePtr) {
}

lifuren::MainWindow::~MainWindow() {
    SPDLOG_DEBUG("关闭MainWindow");
    DELETE_MEDIA_PTR(audio);
    DELETE_MEDIA_PTR(image);
    DELETE_MEDIA_PTR(video);
    DELETE_MEDIA_PTR(poetry);
    DELETE_PTR(aboutButtonPtr);
    DELETE_PTR(aboutWindowPtr);
}

void lifuren::MainWindow::drawElement() {
    // this->w() - 20 - 20 - 10 - 10 - 10 = this->w() - 70
    // 音频
    this->audioGroupPtr = new Fl_Group(0, 10, this->w() / 2 - 20, 30);
    this->audioGcPtr = new Fl_Button(20, 10, (this->w() - 70) / 4, 30, "音频内容生成");
    this->audioTsPtr = new Fl_Button((this->w() - 70) / 4 + 30, 10, (this->w() - 70) / 4, 30, "音频风格转换");
    this->audioGroupPtr->end();
    // 图片
    this->imageGroupPtr = new Fl_Group(this->w() / 2, 10, this->w() / 2 - 20, 30);
    this->imageGcPtr = new Fl_Button((this->w() - 70) / 4 * 2 + 40, 10, (this->w() - 70) / 4, 30, "图片内容生成");
    this->imageTsPtr = new Fl_Button((this->w() - 70) / 4 * 3 + 50, 10, (this->w() - 70) / 4, 30, "图片风格转换");
    this->imageGroupPtr->end();
    // 视频
    this->videoGroupPtr = new Fl_Group(0, 50, this->w() / 2 - 20, 30);
    this->videoGcPtr = new Fl_Button(20, 50, (this->w() - 70) / 4, 30, "视频内容生成");
    this->videoTsPtr = new Fl_Button((this->w() - 70) / 4 + 30, 50, (this->w() - 70) / 4, 30, "视频风格转换");
    this->videoGroupPtr->end();
    // 诗词
    this->poetryGroupPtr = new Fl_Group(this->w() / 2, 50, this->w() / 2 - 20, 30);
    this->poetryGcPtr = new Fl_Button((this->w() - 70) / 4 * 2 + 40, 50, (this->w() - 70) / 4, 30, "诗词内容生成");
    this->poetryTsPtr = new Fl_Button((this->w() - 70) / 4 * 3 + 50, 50, (this->w() - 70) / 4, 30, "诗词风格转换");
    this->poetryGroupPtr->end();
    // 关于
    this->aboutButtonPtr = new Fl_Button((this->w() - 70) / 4 * 3 + 50, this->h() - 40, (this->w() - 70) / 4, 30, "关于");
    CALLBACK_BINDER(about);
    this->resizable(this);
}

void lifuren::MainWindow::imageGc() {
}

void lifuren::MainWindow::about() {
    if(this->aboutWindowPtr != nullptr) {
        this->aboutWindowPtr->show();
        return;
    }
    this->aboutWindowPtr = new AboutWindow(512, 256, "关于");
    this->aboutWindowPtr->init();
    this->aboutWindowPtr->show();
    this->aboutWindowPtr->callback([](Fl_Widget* widgetPtr, void* voidPtr) -> void {
        // 隐藏关于窗口
        widgetPtr->hide();
        // 释放关于窗口
        MainWindow* mainPtr = (MainWindow*) voidPtr;
        delete mainPtr->aboutWindowPtr;
        mainPtr->aboutWindowPtr = nullptr;
    }, this);
}

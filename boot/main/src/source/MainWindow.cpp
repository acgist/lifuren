#include "../header/Window.hpp"

lifuren::MainWindow::MainWindow(int width, int height, const char* titlePtr) : LFRWindow(width, height, titlePtr) {
}

lifuren::MainWindow::~MainWindow() {
    SPDLOG_DEBUG("关闭MainWindow");
    if(this->audioGcPtr != nullptr) {
        delete this->audioGcPtr;
        this->audioGcPtr = nullptr;
    }
    if(this->audioTsPtr != nullptr) {
        delete this->audioTsPtr;
        this->audioTsPtr = nullptr;
    }
    if(this->audioGroupPtr != nullptr) {
        delete this->audioGroupPtr;
        this->audioGroupPtr = nullptr;
    }
    if(this->imageGcPtr != nullptr) {
        delete this->imageGcPtr;
        this->imageGcPtr = nullptr;
    }
    if(this->imageTsPtr != nullptr) {
        delete this->imageTsPtr;
        this->imageTsPtr = nullptr;
    }
    if(this->imageGroupPtr != nullptr) {
        delete this->imageGroupPtr;
        this->imageGroupPtr = nullptr;
    }
    if(this->videoGcPtr != nullptr) {
        delete this->videoGcPtr;
        this->videoGcPtr = nullptr;
    }
    if(this->videoTsPtr != nullptr) {
        delete this->videoTsPtr;
        this->videoTsPtr = nullptr;
    }
    if(this->videoGroupPtr != nullptr) {
        delete this->videoGroupPtr;
        this->videoGroupPtr = nullptr;
    }
    if(this->poetryGcPtr != nullptr) {
        delete this->poetryGcPtr;
        this->poetryGcPtr = nullptr;
    }
    if(this->poetryTsPtr != nullptr) {
        delete this->poetryTsPtr;
        this->poetryTsPtr = nullptr;
    }
    if(this->poetryGroupPtr != nullptr) {
        delete this->poetryGroupPtr;
        this->poetryGroupPtr = nullptr;
    }
    if(this->aboutButtonPtr != nullptr) {
        delete this->aboutButtonPtr;
        this->aboutButtonPtr = nullptr;
    }
    if(this->aboutWindowPtr != nullptr) {
        delete this->aboutWindowPtr;
        this->aboutWindowPtr = nullptr;
    }
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
    this->aboutButtonPtr->callback([](Fl_Widget* widgetPtr, void* voidPtr) -> void {
        ((MainWindow*) voidPtr)->about();
    }, this);
    this->resizable(this);
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

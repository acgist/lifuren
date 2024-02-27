#include "../header/Window.hpp"

// 删除资源指针
#ifndef LFR_DELETE_MEDIA_PTR
#define LFR_DELETE_MEDIA_PTR(mediaType)           \
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
    }
#endif

// 回调按钮绑定
#ifndef LFR_CALLBACK_FUNCTION_BINDER
#define LFR_CALLBACK_FUNCTION_BINDER(button, function)                       \
    this->button->callback([](Fl_Widget* widgetPtr, void* voidPtr) -> void { \
        ((MainWindow*) voidPtr)->function();                                 \
    }, this);
#endif

// 回调函数声明
#ifndef LFR_CALLBACK_FUNCTION
#define LFR_CALLBACK_FUNCTION(methodName, WindowName, windowPtr, width, height)     \
    void lifuren::MainWindow::methodName() {                                        \
        if(this->windowPtr != nullptr) {                                            \
            this->windowPtr->show();                                                \
            return;                                                                 \
        }                                                                           \
        this->windowPtr = new WindowName(width, height);                            \
        this->windowPtr->init();                                                    \
        this->windowPtr->show();                                                    \
        this->windowPtr->callback([](Fl_Widget* widgetPtr, void* voidPtr) -> void { \
            widgetPtr->hide();                                                      \
            MainWindow* mainPtr = (MainWindow*) voidPtr;                            \
            delete mainPtr->windowPtr;                                              \
            mainPtr->windowPtr = nullptr;                                           \
        }, this);                                                                   \
    }
#endif

lifuren::MainWindow::MainWindow(int width, int height, const char* title) : LFRWindow(width, height, title) {
    // TODO：生成目录（config）
    SETTINGS.loadFile(SETTINGS_PATH);
}

lifuren::MainWindow::~MainWindow() {
    SPDLOG_DEBUG("关闭MainWindow");
    LFR_DELETE_MEDIA_PTR(audio);
    LFR_DELETE_MEDIA_PTR(image);
    LFR_DELETE_MEDIA_PTR(video);
    LFR_DELETE_MEDIA_PTR(poetry);
    LFR_DELETE_PTR(aboutButtonPtr);
    LFR_DELETE_PTR(aboutWindowPtr);
}

void lifuren::MainWindow::drawElement() {
    // 音频
    this->audioGcPtr = new Fl_Button(20,                        10, (this->w() - 60) / 2, 30, "音频内容生成");
    this->audioTsPtr = new Fl_Button((this->w() - 60) / 2 + 40, 10, (this->w() - 60) / 2, 30, "音频风格转换");
    // 图片
    this->imageGcPtr = new Fl_Button(20,                        50, (this->w() - 60) / 2, 30, "图片内容生成");
    this->imageTsPtr = new Fl_Button((this->w() - 60) / 2 + 40, 50, (this->w() - 60) / 2, 30, "图片风格转换");
    LFR_CALLBACK_FUNCTION_BINDER(imageGcPtr, imageGc);
    // 视频
    this->videoGcPtr = new Fl_Button(20,                        90, (this->w() - 60) / 2, 30, "视频内容生成");
    this->videoTsPtr = new Fl_Button((this->w() - 60) / 2 + 40, 90, (this->w() - 60) / 2, 30, "视频风格转换");
    // 诗词
    this->poetryGcPtr = new Fl_Button(20,                        130, (this->w() - 60) / 2, 30, "诗词内容生成");
    this->poetryTsPtr = new Fl_Button((this->w() - 60) / 2 + 40, 130, (this->w() - 60) / 2, 30, "诗词风格转换");
    // 关于
    this->aboutButtonPtr = new Fl_Button((this->w() - 80) / 4 * 3 + 60, this->h() - 40, (this->w() - 80) / 4, 30, "关于");
    LFR_CALLBACK_FUNCTION_BINDER(aboutButtonPtr, about);
    this->resizable(this);
}

LFR_CALLBACK_FUNCTION(imageGc, ImageGCWindow, imageGcWindowPtr, 1200, 800);
LFR_CALLBACK_FUNCTION(about, AboutWindow, aboutWindowPtr, 512, 256);

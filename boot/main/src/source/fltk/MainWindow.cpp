#include "lifuren/FLTK.hpp"

#include "lifuren/Lifuren.hpp"

#include "FL/fl_ask.H"
#include "FL/Fl_Shared_Image.H"

#include "spdlog/spdlog.h"

// 回调按钮绑定
#ifndef LFR_BUTTON_CALLBACK_FUNCTION_BINDER
#define LFR_BUTTON_CALLBACK_FUNCTION_BINDER(button, function)                \
    this->button->callback([](Fl_Widget* widgetPtr, void* voidPtr) -> void { \
        ((MainWindow*) voidPtr)->function();                                 \
    }, this);
#endif

// 回调函数声明
#ifndef LFR_BUTTON_CALLBACK_FUNCTION
#define LFR_BUTTON_CALLBACK_FUNCTION(methodName, WindowName, windowPtr, width, height) \
    void lifuren::MainWindow::methodName() {                                           \
        if(this->windowPtr != nullptr) {                                               \
            this->windowPtr->show();                                                   \
            return;                                                                    \
        }                                                                              \
        this->windowPtr = new WindowName(width, height);                               \
        this->windowPtr->init();                                                       \
        this->windowPtr->show();                                                       \
        this->windowPtr->callback([](Fl_Widget* widgetPtr, void* voidPtr) -> void {    \
            widgetPtr->hide();                                                         \
            MainWindow* mainPtr = (MainWindow*) voidPtr;                               \
            delete mainPtr->windowPtr;                                                 \
            mainPtr->windowPtr = nullptr;                                              \
        }, this);                                                                      \
    }
#endif

lifuren::MainWindow::MainWindow(int width, int height, const char* title) : Window(width, height, title) {
    // 注册图片
    fl_register_images();
}

lifuren::MainWindow::~MainWindow() {
    SPDLOG_DEBUG("关闭窗口：{}", __FILE__);
    LFR_DELETE_THIS_PTR(docsMarkButtonPtr);
    LFR_DELETE_THIS_PTR(docsMarkWindowPtr);
    LFR_DELETE_THIS_PTR(imageMarkButtonPtr);
    LFR_DELETE_THIS_PTR(imageMarkWindowPtr);
    LFR_DELETE_THIS_PTR(poetryMarkButtonPtr);
    LFR_DELETE_THIS_PTR(poetryMarkWindowPtr);
    LFR_DELETE_THIS_PTR(finetuneButtonPtr);
    LFR_DELETE_THIS_PTR(finetuneWindowPtr);
    LFR_DELETE_THIS_PTR(quantizationButtonPtr);
    LFR_DELETE_THIS_PTR(quantizationWindowPtr);
    LFR_DELETE_THIS_PTR(chatButtonPtr);
    LFR_DELETE_THIS_PTR(chatWindowPtr);
    LFR_DELETE_THIS_PTR(chatWindowPtr);
    LFR_DELETE_THIS_PTR(imageWindowPtr);
    LFR_DELETE_THIS_PTR(videoButtonPtr);
    LFR_DELETE_THIS_PTR(videoWindowPtr);
    LFR_DELETE_THIS_PTR(reloadButtonPtr);
    LFR_DELETE_THIS_PTR(aboutButtonPtr);
    LFR_DELETE_THIS_PTR(aboutWindowPtr);
}

void lifuren::MainWindow::drawElement() {
    // 标记
    this->docsMarkButtonPtr   = new Fl_Button(20,                      10, LFR_HALF_WIDTH(60), 30, "文档标记");
    this->imageMarkButtonPtr  = new Fl_Button(LFR_HALF_WIDTH(60) + 40, 10, LFR_HALF_WIDTH(60), 30, "图片标记");
    this->poetryMarkButtonPtr = new Fl_Button(20,                      50, LFR_HALF_WIDTH(60), 30, "诗词标记");
    // 模型优化
    this->finetuneButtonPtr     = new Fl_Button(20,                      90, LFR_HALF_WIDTH(60), 30, "模型微调");
    this->quantizationButtonPtr = new Fl_Button(LFR_HALF_WIDTH(60) + 40, 90, LFR_HALF_WIDTH(60), 30, "模型量化");
    // 模型
    this->chatButtonPtr  = new Fl_Button(20,                      130, LFR_HALF_WIDTH(60), 30, "聊天");
    this->imageButtonPtr = new Fl_Button(20,                      170, LFR_HALF_WIDTH(60), 30, "图片生成");
    this->videoButtonPtr = new Fl_Button(LFR_HALF_WIDTH(60) + 40, 170, LFR_HALF_WIDTH(60), 30, "视频生成");
    // 重新加载配置
    this->reloadButtonPtr = new Fl_Button((this->w() - 80) / 4 * 2 + 40, this->h() - 40, (this->w() - 80) / 4, 30, "重新加载配置");
    // 关于
    this->aboutButtonPtr  = new Fl_Button((this->w() - 80) / 4 * 3 + 60, this->h() - 40, (this->w() - 80) / 4, 30, "关于");
    // 大小修改
    this->resizable(this);
    // 绑定事件
    LFR_BUTTON_CALLBACK_FUNCTION_BINDER(docsMarkButtonPtr,     docsMark);
    LFR_BUTTON_CALLBACK_FUNCTION_BINDER(imageMarkButtonPtr,    imageMark);
    LFR_BUTTON_CALLBACK_FUNCTION_BINDER(poetryMarkButtonPtr,   poetryMark);
    LFR_BUTTON_CALLBACK_FUNCTION_BINDER(finetuneButtonPtr,     finetune);
    LFR_BUTTON_CALLBACK_FUNCTION_BINDER(quantizationButtonPtr, quantization);
    LFR_BUTTON_CALLBACK_FUNCTION_BINDER(chatButtonPtr,         chat);
    LFR_BUTTON_CALLBACK_FUNCTION_BINDER(imageButtonPtr,        image);
    LFR_BUTTON_CALLBACK_FUNCTION_BINDER(videoButtonPtr,        video);
    LFR_BUTTON_CALLBACK_FUNCTION_BINDER(aboutButtonPtr,        about);
    // 重新加载配置
    this->reloadButtonPtr->callback([](Fl_Widget*, void*) {
        lifuren::loadConfig();
    }, this);
}

// 定义窗口
LFR_BUTTON_CALLBACK_FUNCTION(docsMark,     DocsMarkWindow,     docsMarkWindowPtr,     1200, 800);
LFR_BUTTON_CALLBACK_FUNCTION(imageMark,    ImageMarkWindow,    imageMarkWindowPtr,    1200, 800);
LFR_BUTTON_CALLBACK_FUNCTION(poetryMark,   PoetryMarkWindow,   poetryMarkWindowPtr,   1200, 800);
LFR_BUTTON_CALLBACK_FUNCTION(finetune,     FinetuneWindow,     finetuneWindowPtr,     1200, 800);
LFR_BUTTON_CALLBACK_FUNCTION(quantization, QuantizationWindow, quantizationWindowPtr, 1200, 800);
LFR_BUTTON_CALLBACK_FUNCTION(chat,         ChatWindow,         chatWindowPtr,         1200, 800);
LFR_BUTTON_CALLBACK_FUNCTION(image,        ImageWindow,        imageWindowPtr,        1200, 800);
LFR_BUTTON_CALLBACK_FUNCTION(video,        VideoWindow,        videoWindowPtr,        1200, 800);
LFR_BUTTON_CALLBACK_FUNCTION(about,        AboutWindow,        aboutWindowPtr,        512,  256);

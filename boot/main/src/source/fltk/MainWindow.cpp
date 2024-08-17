#include "lifuren/FLTK.hpp"

#include "spdlog/spdlog.h"

#include "lifuren/Lifuren.hpp"

#include "FL/fl_ask.H"
#include "FL/Fl_Input.H"
#include "FL/Fl_Button.H"
#include "FL/Fl_Shared_Image.H"

static Fl_Button* imageMarkButtonPtr    { nullptr };
static Fl_Button* poetryMarkButtonPtr   { nullptr };
static Fl_Button* documentMarkButtonPtr { nullptr };
static Fl_Button* finetuneButtonPtr     { nullptr };
static Fl_Button* quantizationButtonPtr { nullptr };
static Fl_Button* chatButtonPtr  { nullptr };
static Fl_Button* imageButtonPtr { nullptr };
static Fl_Button* poetryButtonPtr{ nullptr };
static Fl_Button* aboutButtonPtr { nullptr };
static Fl_Button* reloadButtonPtr{ nullptr };

// 窗口宽度
#ifndef LFR_HALF_WIDTH
#define LFR_HALF_WIDTH(padding) (this->w() - padding) / 2
#endif

// 回调按钮绑定
#ifndef LFR_BUTTON_CALLBACK_FUNCTION_BINDER
#define LFR_BUTTON_CALLBACK_FUNCTION_BINDER(button, function)          \
    button->callback([](Fl_Widget* widgetPtr, void* voidPtr) -> void { \
        ((MainWindow*) voidPtr)->function();                           \
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
    LFR_DELETE_PTR(imageMarkButtonPtr);
    LFR_DELETE_THIS_PTR(imageMarkWindowPtr);
    LFR_DELETE_PTR(poetryMarkButtonPtr);
    LFR_DELETE_THIS_PTR(poetryMarkWindowPtr);
    LFR_DELETE_PTR(documentMarkButtonPtr);
    LFR_DELETE_THIS_PTR(documentMarkWindowPtr);
    LFR_DELETE_PTR(finetuneButtonPtr);
    LFR_DELETE_THIS_PTR(finetuneWindowPtr);
    LFR_DELETE_PTR(quantizationButtonPtr);
    LFR_DELETE_THIS_PTR(quantizationWindowPtr);
    LFR_DELETE_PTR(chatButtonPtr);
    LFR_DELETE_THIS_PTR(chatWindowPtr);
    LFR_DELETE_PTR(imageButtonPtr);
    LFR_DELETE_THIS_PTR(imageWindowPtr);
    LFR_DELETE_PTR(poetryButtonPtr);
    LFR_DELETE_THIS_PTR(poetryWindowPtr);
    LFR_DELETE_PTR(aboutButtonPtr);
    LFR_DELETE_THIS_PTR(aboutWindowPtr);
    LFR_DELETE_PTR(reloadButtonPtr);
}

void lifuren::MainWindow::drawElement() {
    // 数据标记
    imageMarkButtonPtr    = new Fl_Button(20,                      10, LFR_HALF_WIDTH(60), 30, "图片标记");
    poetryMarkButtonPtr   = new Fl_Button(LFR_HALF_WIDTH(60) + 40, 10, LFR_HALF_WIDTH(60), 30, "诗词标记");
    documentMarkButtonPtr = new Fl_Button(20,                      50, LFR_HALF_WIDTH(60), 30, "文档标记");
    // 模型优化
    finetuneButtonPtr     = new Fl_Button(20,                      90, LFR_HALF_WIDTH(60), 30, "模型微调");
    quantizationButtonPtr = new Fl_Button(LFR_HALF_WIDTH(60) + 40, 90, LFR_HALF_WIDTH(60), 30, "模型量化");
    // 模型功能
    chatButtonPtr   = new Fl_Button(20,                      130, LFR_HALF_WIDTH(60), 30, "聊天");
    imageButtonPtr  = new Fl_Button(20,                      170, LFR_HALF_WIDTH(60), 30, "图片生成");
    poetryButtonPtr = new Fl_Button(LFR_HALF_WIDTH(60) + 40, 170, LFR_HALF_WIDTH(60), 30, "诗词生成");
    // 关于
    aboutButtonPtr  = new Fl_Button(this->w() - 100, this->h() - 40, 80,  30, "关于");
    // 重新加载配置
    reloadButtonPtr = new Fl_Button(this->w() - 260, this->h() - 40, 140, 30, "重新加载配置");
    // 大小修改
    this->resizable(this);
    // 绑定事件
    LFR_BUTTON_CALLBACK_FUNCTION_BINDER(imageMarkButtonPtr,    imageMark);
    LFR_BUTTON_CALLBACK_FUNCTION_BINDER(poetryMarkButtonPtr,   poetryMark);
    LFR_BUTTON_CALLBACK_FUNCTION_BINDER(documentMarkButtonPtr, documentMark);
    LFR_BUTTON_CALLBACK_FUNCTION_BINDER(finetuneButtonPtr,     finetune);
    LFR_BUTTON_CALLBACK_FUNCTION_BINDER(quantizationButtonPtr, quantization);
    LFR_BUTTON_CALLBACK_FUNCTION_BINDER(chatButtonPtr,         chat);
    LFR_BUTTON_CALLBACK_FUNCTION_BINDER(imageButtonPtr,        image);
    LFR_BUTTON_CALLBACK_FUNCTION_BINDER(poetryButtonPtr,       poetry);
    LFR_BUTTON_CALLBACK_FUNCTION_BINDER(aboutButtonPtr,        about);
    // 重新加载配置
    reloadButtonPtr->callback([](Fl_Widget*, void*) {
        lifuren::loadConfig();
    }, this);
}

// 定义窗口
LFR_BUTTON_CALLBACK_FUNCTION(imageMark,    ImageMarkWindow,    imageMarkWindowPtr,    LFR_WINDOW_WIDTH, LFR_WINDOW_HEIGHT);
LFR_BUTTON_CALLBACK_FUNCTION(poetryMark,   PoetryMarkWindow,   poetryMarkWindowPtr,   LFR_WINDOW_WIDTH, LFR_WINDOW_HEIGHT);
LFR_BUTTON_CALLBACK_FUNCTION(documentMark, DocumentMarkWindow, documentMarkWindowPtr, LFR_WINDOW_WIDTH_CONFIG, LFR_WINDOW_HEIGHT_CONFIG);
LFR_BUTTON_CALLBACK_FUNCTION(finetune,     FinetuneWindow,     finetuneWindowPtr,     LFR_WINDOW_WIDTH, LFR_WINDOW_HEIGHT);
LFR_BUTTON_CALLBACK_FUNCTION(quantization, QuantizationWindow, quantizationWindowPtr, LFR_WINDOW_WIDTH, LFR_WINDOW_HEIGHT);
LFR_BUTTON_CALLBACK_FUNCTION(chat,         ChatWindow,         chatWindowPtr,         LFR_WINDOW_WIDTH, LFR_WINDOW_HEIGHT);
LFR_BUTTON_CALLBACK_FUNCTION(image,        ImageWindow,        imageWindowPtr,        LFR_WINDOW_WIDTH, LFR_WINDOW_HEIGHT);
LFR_BUTTON_CALLBACK_FUNCTION(poetry,       PoetryWindow,       poetryWindowPtr,       LFR_WINDOW_WIDTH, LFR_WINDOW_HEIGHT);
LFR_BUTTON_CALLBACK_FUNCTION(about,        AboutWindow,        aboutWindowPtr,        512,  256);

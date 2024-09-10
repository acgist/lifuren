#include "lifuren/FLTK.hpp"

#include "spdlog/spdlog.h"

#include "lifuren/Lifuren.hpp"

#include "FL/fl_ask.H"
#include "FL/filename.H"
#include "FL/Fl_Input.H"
#include "FL/Fl_Button.H"
#include "FL/Fl_Shared_Image.H"

static Fl_Button* markButtonPtr  { nullptr };
static Fl_Button* imageButtonPtr { nullptr };
static Fl_Button* poetryButtonPtr{ nullptr };
static Fl_Button* aboutButtonPtr { nullptr };
static Fl_Button* reloadButtonPtr{ nullptr };
static Fl_Button* finetuneButtonPtr     { nullptr };
static Fl_Button* quantizationButtonPtr { nullptr };

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
    LFR_DELETE_PTR(markButtonPtr);
    LFR_DELETE_THIS_PTR(markWindowPtr);
    LFR_DELETE_PTR(imageButtonPtr);
    LFR_DELETE_THIS_PTR(imageWindowPtr);
    LFR_DELETE_PTR(poetryButtonPtr);
    LFR_DELETE_THIS_PTR(poetryWindowPtr);
    LFR_DELETE_PTR(aboutButtonPtr);
    LFR_DELETE_THIS_PTR(aboutWindowPtr);
    LFR_DELETE_PTR(reloadButtonPtr);
    LFR_DELETE_PTR(finetuneButtonPtr);
    LFR_DELETE_PTR(quantizationButtonPtr);
}

void lifuren::MainWindow::drawElement() {
    // 绘制界面
    markButtonPtr         = new Fl_Button(20,                      10,  this->w() - 40,     80, "诗词标记");
    imageButtonPtr        = new Fl_Button(20,                      100, LFR_HALF_WIDTH(60), 80, "图片生成");
    poetryButtonPtr       = new Fl_Button(LFR_HALF_WIDTH(60) + 40, 100, LFR_HALF_WIDTH(60), 80, "诗词生成");
    finetuneButtonPtr     = new Fl_Button(20,              this->h() - 40, 120, 30, "模型微调");
    quantizationButtonPtr = new Fl_Button(140,             this->h() - 40, 120, 30, "模型量化");
    reloadButtonPtr       = new Fl_Button(this->w() - 260, this->h() - 40, 120, 30, "加载配置");
    aboutButtonPtr        = new Fl_Button(this->w() - 140, this->h() - 40, 120, 30, "关于项目");
    // 大小修改
    this->resizable(this);
    // 绑定事件
    LFR_BUTTON_CALLBACK_FUNCTION_BINDER(markButtonPtr,   mark);
    LFR_BUTTON_CALLBACK_FUNCTION_BINDER(imageButtonPtr,  image);
    LFR_BUTTON_CALLBACK_FUNCTION_BINDER(poetryButtonPtr, poetry);
    LFR_BUTTON_CALLBACK_FUNCTION_BINDER(aboutButtonPtr,  about);
    // 模型微调
    finetuneButtonPtr->callback([](Fl_Widget*, void*) {
        const int ret = fl_open_uri("https://gitee.com/acgist/lifuren/tree/master/docs/optimize");
        SPDLOG_DEBUG("打开模型微调：{}", ret);
    }, this);
    // 模型量化
    quantizationButtonPtr->callback([](Fl_Widget*, void*) {
        const int ret = fl_open_uri("https://gitee.com/acgist/lifuren/tree/master/docs/optimize");
        SPDLOG_DEBUG("打开模型量化：{}", ret);
    }, this);
    // 加载配置
    reloadButtonPtr->callback([](Fl_Widget*, void*) {
        lifuren::loadConfig();
    }, this);
}

// 定义窗口
LFR_BUTTON_CALLBACK_FUNCTION(mark,   MarkWindow,   markWindowPtr,   LFR_WINDOW_WIDTH, LFR_WINDOW_HEIGHT);
LFR_BUTTON_CALLBACK_FUNCTION(image,  ImageWindow,  imageWindowPtr,  LFR_WINDOW_WIDTH, LFR_WINDOW_HEIGHT);
LFR_BUTTON_CALLBACK_FUNCTION(poetry, PoetryWindow, poetryWindowPtr, LFR_WINDOW_WIDTH, LFR_WINDOW_HEIGHT);
LFR_BUTTON_CALLBACK_FUNCTION(about,  AboutWindow,  aboutWindowPtr,  512,              256);

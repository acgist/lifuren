#include "lifuren/FLTK.hpp"

#include "spdlog/spdlog.h"

#include "lifuren/Ptr.hpp"
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

static lifuren::MarkWindow  * markWindowPtr  { nullptr };
static lifuren::ImageWindow * imageWindowPtr { nullptr };
static lifuren::PoetryWindow* poetryWindowPtr{ nullptr };
static lifuren::AboutWindow * aboutWindowPtr { nullptr };

static void markCallback  (Fl_Widget*, void*);
static void imageCallback (Fl_Widget*, void*);
static void poetryCallback(Fl_Widget*, void*);
static void aboutCallback (Fl_Widget*, void*);

// 窗口宽度
#ifndef LFR_HALF_WIDTH
#define LFR_HALF_WIDTH(padding) (this->w() - padding) / 2
#endif

// 回调函数声明
#ifndef LFR_BUTTON_CALLBACK_FUNCTION
#define LFR_BUTTON_CALLBACK_FUNCTION(methodName, WindowName, windowPtr, width, height) \
    static void methodName(Fl_Widget*, void* voidPtr) {                                \
        if(windowPtr != nullptr) {                                                     \
            windowPtr->show();                                                         \
            return;                                                                    \
        }                                                                              \
        windowPtr = new lifuren::WindowName(width, height);                            \
        windowPtr->init();                                                             \
        windowPtr->show();                                                             \
        windowPtr->callback([](Fl_Widget* widgetPtr, void*) -> void {                  \
            widgetPtr->hide();                                                         \
            delete windowPtr;                                                          \
            windowPtr = nullptr;                                                       \
        }, voidPtr);                                                                   \
    }
#endif

lifuren::MainWindow::MainWindow(int width, int height, const char* title) : Window(width, height, title) {
    // 注册图片
    fl_register_images();
}

lifuren::MainWindow::~MainWindow() {
    LFR_DELETE_PTR(markButtonPtr);
    LFR_DELETE_PTR(markWindowPtr);
    LFR_DELETE_PTR(imageButtonPtr);
    LFR_DELETE_PTR(imageWindowPtr);
    LFR_DELETE_PTR(poetryButtonPtr);
    LFR_DELETE_PTR(poetryWindowPtr);
    LFR_DELETE_PTR(aboutButtonPtr);
    LFR_DELETE_PTR(aboutWindowPtr);
    LFR_DELETE_PTR(reloadButtonPtr);
    LFR_DELETE_PTR(finetuneButtonPtr);
    LFR_DELETE_PTR(quantizationButtonPtr);
}

void lifuren::MainWindow::drawElement() {
    // 绘制界面
    markButtonPtr         = new Fl_Button(20,                      10,             this->w() - 40,     80, "诗词标记");
    imageButtonPtr        = new Fl_Button(20,                      100,            LFR_HALF_WIDTH(60), 80, "图片生成");
    poetryButtonPtr       = new Fl_Button(LFR_HALF_WIDTH(60) + 40, 100,            LFR_HALF_WIDTH(60), 80, "诗词生成");
    aboutButtonPtr        = new Fl_Button(this->w() - 140,         this->h() - 40, 120,                30, "关于项目");
    reloadButtonPtr       = new Fl_Button(this->w() - 260,         this->h() - 40, 120,                30, "加载配置");
    finetuneButtonPtr     = new Fl_Button(20,                      this->h() - 40, 120,                30, "模型微调");
    quantizationButtonPtr = new Fl_Button(140,                     this->h() - 40, 120,                30, "模型量化");
    // 大小修改
    this->resizable(this);
    // 绑定事件
    markButtonPtr->callback(markCallback, this);
    imageButtonPtr->callback(imageCallback, this);
    poetryButtonPtr->callback(poetryCallback, this);
    aboutButtonPtr->callback(aboutCallback, this);
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
LFR_BUTTON_CALLBACK_FUNCTION(markCallback,   MarkWindow,   markWindowPtr,   LFR_WINDOW_WIDTH, LFR_WINDOW_HEIGHT);
LFR_BUTTON_CALLBACK_FUNCTION(imageCallback,  ImageWindow,  imageWindowPtr,  LFR_WINDOW_WIDTH, LFR_WINDOW_HEIGHT);
LFR_BUTTON_CALLBACK_FUNCTION(poetryCallback, PoetryWindow, poetryWindowPtr, LFR_WINDOW_WIDTH, LFR_WINDOW_HEIGHT);
LFR_BUTTON_CALLBACK_FUNCTION(aboutCallback,  AboutWindow,  aboutWindowPtr,  512,              256);

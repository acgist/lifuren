/**
 * FLTK主界面
 * 
 * 主要功能入口
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#include "lifuren/FLTK.hpp"

#include "lifuren/Raii.hpp"
#include "lifuren/Config.hpp"

#include "FL/Fl_Button.H"
#include "FL/Fl_Shared_Image.H"

static Fl_Button* audioButtonPtr { nullptr };
static Fl_Button* videoButtonPtr { nullptr };
static Fl_Button* poetryButtonPtr{ nullptr };
static Fl_Button* aboutButtonPtr { nullptr };
static Fl_Button* reloadButtonPtr{ nullptr };

static lifuren::AudioWindow * audioWindowPtr { nullptr };
static lifuren::VideoWindow * videoWindowPtr { nullptr };
static lifuren::PoetryWindow* poetryWindowPtr{ nullptr };
static lifuren::AboutWindow * aboutWindowPtr { nullptr };

static void audioCallback (Fl_Widget*, void*);
static void videoCallback (Fl_Widget*, void*);
static void poetryCallback(Fl_Widget*, void*);
static void aboutCallback (Fl_Widget*, void*);
static void reloadCallback(Fl_Widget*, void*);

// 窗口宽度
#ifndef LFR_HALF_WIDTH
#define LFR_HALF_WIDTH(padding) (this->w() - padding) / 2
#endif

// 回调函数
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
    LFR_DELETE_PTR(audioButtonPtr);
    LFR_DELETE_PTR(audioWindowPtr);
    LFR_DELETE_PTR(videoButtonPtr);
    LFR_DELETE_PTR(videoWindowPtr);
    LFR_DELETE_PTR(poetryButtonPtr);
    LFR_DELETE_PTR(poetryWindowPtr);
    LFR_DELETE_PTR(aboutButtonPtr);
    LFR_DELETE_PTR(aboutWindowPtr);
    LFR_DELETE_PTR(reloadButtonPtr);
}

void lifuren::MainWindow::drawElement() {
    // 绘制界面
    audioButtonPtr  = new Fl_Button(20,                      10,             LFR_HALF_WIDTH(60), 80, "音频生成");
    videoButtonPtr  = new Fl_Button(LFR_HALF_WIDTH(60) + 40, 10,             LFR_HALF_WIDTH(60), 80, "视频生成");
    poetryButtonPtr = new Fl_Button(20,                      100,            this->w() - 40,     80, "诗词生成");
    reloadButtonPtr = new Fl_Button(this->w() - 260,         this->h() - 40, 120,                30, "加载配置");
    aboutButtonPtr  = new Fl_Button(this->w() - 140,         this->h() - 40, 120,                30, "关于项目");
    // 大小修改
    this->resizable(this);
    // 绑定事件
    audioButtonPtr ->callback(audioCallback , this);
    videoButtonPtr ->callback(videoCallback , this);
    poetryButtonPtr->callback(poetryCallback, this);
    aboutButtonPtr ->callback(aboutCallback , this);
    reloadButtonPtr->callback(reloadCallback, this);
}

// 定义窗口
LFR_BUTTON_CALLBACK_FUNCTION(audioCallback,  AudioWindow,  audioWindowPtr,  LFR_WINDOW_WIDTH / 2, LFR_WINDOW_HEIGHT / 2);
LFR_BUTTON_CALLBACK_FUNCTION(videoCallback,  VideoWindow,  videoWindowPtr,  LFR_WINDOW_WIDTH / 2, LFR_WINDOW_HEIGHT / 2);
LFR_BUTTON_CALLBACK_FUNCTION(poetryCallback, PoetryWindow, poetryWindowPtr, LFR_WINDOW_WIDTH / 2, LFR_WINDOW_HEIGHT / 2);
LFR_BUTTON_CALLBACK_FUNCTION(aboutCallback,  AboutWindow,  aboutWindowPtr,  LFR_WINDOW_WIDTH / 2, LFR_WINDOW_HEIGHT / 2);

static void reloadCallback(Fl_Widget*, void*) {
    lifuren::config::loadConfig();
}

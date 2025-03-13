#include "lifuren/FLTK.hpp"

#include "lifuren/Raii.hpp"
#include "lifuren/Config.hpp"

#include "FL/Fl_Button.H"
#include "FL/Fl_Text_Buffer.H"
#include "FL/Fl_Text_Display.H"
#include "FL/Fl_Shared_Image.H"

static Fl_Button* bachButtonPtr      { nullptr };
static Fl_Button* chopinButtonPtr    { nullptr };
static Fl_Button* mozartButtonPtr    { nullptr };
static Fl_Button* wudaoziButtonPtr   { nullptr };
static Fl_Button* shikuangButtonPtr  { nullptr };
static Fl_Button* musicScoreButtonPtr{ nullptr };
static Fl_Button* aboutButtonPtr     { nullptr };
static Fl_Button* configButtonPtr    { nullptr };
static Fl_Text_Display* displayPtr   { nullptr };
static Fl_Text_Buffer * bufferPtr    { nullptr };

static lifuren::AboutWindow     * aboutWindowPtr     { nullptr };
static lifuren::ConfigWindow    * configWindowPtr    { nullptr };
static lifuren::MusicScoreWindow* musicScoreWindowPtr{ nullptr };

static void bachCallback      (Fl_Widget*, void*);
static void chopinCallback    (Fl_Widget*, void*);
static void mozartCallback    (Fl_Widget*, void*);
static void wudaoziCallback   (Fl_Widget*, void*);
static void shikuangCallback  (Fl_Widget*, void*);
static void musicScoreCallback(Fl_Widget*, void*);
static void aboutCallback     (Fl_Widget*, void*);
static void configCallback    (Fl_Widget*, void*);

// 窗口函数
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
    fl_register_images();
}

lifuren::MainWindow::~MainWindow() {
    LFR_DELETE_PTR(bachButtonPtr      );
    LFR_DELETE_PTR(chopinButtonPtr    );
    LFR_DELETE_PTR(mozartButtonPtr    );
    LFR_DELETE_PTR(wudaoziButtonPtr   );
    LFR_DELETE_PTR(shikuangButtonPtr  );
    LFR_DELETE_PTR(musicScoreButtonPtr);
    LFR_DELETE_PTR(aboutButtonPtr     );
    LFR_DELETE_PTR(configButtonPtr    );
    LFR_DELETE_PTR(displayPtr         );
    LFR_DELETE_PTR(bufferPtr          );
}

void lifuren::MainWindow::drawElement() {
    bachButtonPtr       = new Fl_Button(                           20,  10, (this->w() - 60) / 3, 80, "音频识谱");
    chopinButtonPtr     = new Fl_Button((this->w() - 60) / 3     + 30,  10, (this->w() - 60) / 3, 80, "简谱识谱");
    mozartButtonPtr     = new Fl_Button((this->w() - 60) / 3 * 2 + 40,  10, (this->w() - 60) / 3, 80, "五线谱识谱");
    shikuangButtonPtr   = new Fl_Button(                           20, 100, (this->w() - 50) / 2, 80, "音频风格迁移");
    wudaoziButtonPtr    = new Fl_Button((this->w() - 50) / 2     + 30, 100, (this->w() - 50) / 2, 80, "图片风格迁移");
    musicScoreButtonPtr = new Fl_Button(                           20, 190, (this->w() - 40)    , 80, "乐谱查看");
    configButtonPtr     = new Fl_Button(                           20, 280, (this->w() - 50) / 2, 80, "配置");
    aboutButtonPtr      = new Fl_Button((this->w() - 50) / 2     + 30, 280, (this->w() - 50) / 2, 80, "关于");
    displayPtr          = new Fl_Text_Display(20, 40, this->w() - 40, this->h() - 90, "关于");
    bufferPtr           = new Fl_Text_Buffer();
    displayPtr->begin();
    displayPtr->color(FL_BACKGROUND_COLOR);
    displayPtr->buffer(bufferPtr);
    displayPtr->wrap_mode(displayPtr->WRAP_AT_COLUMN, displayPtr->textfont());
    displayPtr->end();
}

void lifuren::MainWindow::bindEvent() {
    aboutButtonPtr ->callback(aboutCallback,  this);
    configButtonPtr->callback(configCallback, this);
}

LFR_BUTTON_CALLBACK_FUNCTION(aboutCallback,  AboutWindow,  aboutWindowPtr,  LFR_DIALOG_WIDTH, LFR_DIALOG_HEIGHT);
LFR_BUTTON_CALLBACK_FUNCTION(configCallback, ConfigWindow, configWindowPtr, LFR_DIALOG_WIDTH, LFR_DIALOG_HEIGHT);

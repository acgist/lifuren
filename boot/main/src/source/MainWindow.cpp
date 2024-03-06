#include "../header/Window.hpp"

#include "Label.hpp"
#include "Jsons.hpp"

// 删除资源指针
#ifndef LFR_DELETE_MODEL_PTR
#define LFR_DELETE_MODEL_PTR(modelType)           \
    SPDLOG_DEBUG("delete " #modelType " ptr");    \
    if(this->modelType##GcPtr != nullptr) {       \
        delete this->modelType##GcPtr;            \
        this->modelType##GcPtr = nullptr;         \
    }                                             \
    if(this->modelType##TsPtr != nullptr) {       \
        delete this->modelType##TsPtr;            \
        this->modelType##TsPtr = nullptr;         \
    }                                             \
    if(this->modelType##GcWindowPtr != nullptr) { \
        delete this->modelType##GcWindowPtr;      \
        this->modelType##GcWindowPtr = nullptr;   \
    }                                             \
    if(this->modelType##TsWindowPtr != nullptr) { \
        delete this->modelType##TsWindowPtr;      \
        this->modelType##TsWindowPtr = nullptr;   \
    }
#endif

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

// 重新加载配置
static void reload(Fl_Widget*, void*);
// 禁用按钮提示
static void disable(Fl_Widget*, void*);

lifuren::MainWindow::MainWindow(int width, int height, const char* title) : LFRWindow(width, height, title) {
    // 注册图片
    fl_register_images();
    // 加载配置
    reload(nullptr, nullptr);
}

lifuren::MainWindow::~MainWindow() {
    SPDLOG_DEBUG("关闭窗口：{}", __FILE__);
    LFR_DELETE_MODEL_PTR(audio);
    LFR_DELETE_MODEL_PTR(image);
    LFR_DELETE_MODEL_PTR(video);
    LFR_DELETE_MODEL_PTR(poetry);
    LFR_DELETE_THIS_PTR(reloadButtonPtr);
    LFR_DELETE_THIS_PTR(aboutButtonPtr);
    LFR_DELETE_THIS_PTR(aboutWindowPtr);
}

void lifuren::MainWindow::drawElement() {
    // 音频
    this->audioGcPtr = new Fl_Button(20,                        10, (this->w() - 60) / 2, 30, "音频内容生成");
    this->audioTsPtr = new Fl_Button((this->w() - 60) / 2 + 40, 10, (this->w() - 60) / 2, 30, "音频风格转换");
    this->audioGcPtr->callback(disable, this);
    // 图片
    this->imageGcPtr = new Fl_Button(20,                        50, (this->w() - 60) / 2, 30, "图片内容生成");
    this->imageTsPtr = new Fl_Button((this->w() - 60) / 2 + 40, 50, (this->w() - 60) / 2, 30, "图片风格转换");
    // 视频
    this->videoGcPtr = new Fl_Button(20,                        90, (this->w() - 60) / 2, 30, "视频内容生成");
    this->videoTsPtr = new Fl_Button((this->w() - 60) / 2 + 40, 90, (this->w() - 60) / 2, 30, "视频风格转换");
    // 诗词
    this->poetryGcPtr = new Fl_Button(20,                        130, (this->w() - 60) / 2, 30, "诗词内容生成");
    this->poetryTsPtr = new Fl_Button((this->w() - 60) / 2 + 40, 130, (this->w() - 60) / 2, 30, "诗词风格转换");
    this->poetryTsPtr->callback(disable, this);
    // 重新加载配置
    this->reloadButtonPtr = new Fl_Button((this->w() - 80) / 4 * 2 + 40, this->h() - 40, (this->w() - 80) / 4, 30, "重新加载配置");
    // 关于
    this->aboutButtonPtr = new Fl_Button((this->w() - 80) / 4 * 3 + 60, this->h() - 40, (this->w() - 80) / 4, 30, "关于");
    this->resizable(this);
    // 绑定事件
    LFR_BUTTON_CALLBACK_FUNCTION_BINDER(audioTsPtr, audioTs);
    LFR_BUTTON_CALLBACK_FUNCTION_BINDER(imageGcPtr, imageGc);
    LFR_BUTTON_CALLBACK_FUNCTION_BINDER(imageTsPtr, imageTs);
    LFR_BUTTON_CALLBACK_FUNCTION_BINDER(videoGcPtr, videoGc);
    LFR_BUTTON_CALLBACK_FUNCTION_BINDER(videoTsPtr, videoTs);
    LFR_BUTTON_CALLBACK_FUNCTION_BINDER(poetryGcPtr, poetryGc);
    LFR_BUTTON_CALLBACK_FUNCTION_BINDER(aboutButtonPtr, about);
    this->reloadButtonPtr->callback(reload, this);
}

LFR_BUTTON_CALLBACK_FUNCTION(audioTs, AudioTSWindow, audioTsWindowPtr, 1200, 800);
LFR_BUTTON_CALLBACK_FUNCTION(imageGc, ImageGCWindow, imageGcWindowPtr, 1200, 800);
LFR_BUTTON_CALLBACK_FUNCTION(imageTs, ImageTSWindow, imageTsWindowPtr, 1200, 800);
LFR_BUTTON_CALLBACK_FUNCTION(videoGc, VideoGCWindow, videoGcWindowPtr, 1200, 800);
LFR_BUTTON_CALLBACK_FUNCTION(videoTs, VideoTSWindow, videoTsWindowPtr, 1200, 800);
LFR_BUTTON_CALLBACK_FUNCTION(poetryGc, PoetryGCWindow, poetryGcWindowPtr, 1200, 800);
LFR_BUTTON_CALLBACK_FUNCTION(about, AboutWindow, aboutWindowPtr, 512, 256);

static void disable(Fl_Widget* widgetPtr, void* voidPtr) {
    fl_message("功能没有实现");
}

static void reload(Fl_Widget* widgetPtr, void* voidPtr) {
    SPDLOG_DEBUG("重新加载配置");
    lifuren::SETTINGS.clear();
    lifuren::LABEL_AUDIO.clear();
    lifuren::LABEL_IMAGE.clear();
    lifuren::LABEL_VIDEO.clear();
    lifuren::LABEL_POETRY.clear();
    auto settings = lifuren::jsons::loadFile<std::map<std::string, lifuren::Setting>>(lifuren::SETTINGS_PATH);
    lifuren::SETTINGS.insert(settings.begin(), settings.end());
    auto audio = lifuren::LabelFile::loadFile(lifuren::LABEL_AUDIO_PATH);
    lifuren::LABEL_AUDIO.insert(audio.begin(), audio.end());
    auto image = lifuren::LabelFile::loadFile(lifuren::LABEL_IMAGE_PATH);
    lifuren::LABEL_IMAGE.insert(image.begin(), image.end());
    auto video = lifuren::LabelFile::loadFile(lifuren::LABEL_VIDEO_PATH);
    lifuren::LABEL_VIDEO.insert(video.begin(), video.end());
    auto poetry = lifuren::LabelText::loadFile(lifuren::LABEL_POETRY_PATH);
    lifuren::LABEL_POETRY.insert(poetry.begin(), poetry.end());
}

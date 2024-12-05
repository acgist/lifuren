#include "lifuren/FLTK.hpp"

#include <mutex>
#include <thread>

#include "spdlog/spdlog.h"

#include "FL/fl_ask.H"
#include "FL/Fl_Input.H"
#include "FL/Fl_Button.H"
#include "FL/Fl_Choice.H"

#include "lifuren/Raii.hpp"
#include "lifuren/Config.hpp"
#include "lifuren/audio/ComposeClient.hpp"

static Fl_Choice* clientPtr      { nullptr };
static Fl_Input * pathPathPtr    { nullptr };
static Fl_Button* pathChoosePtr  { nullptr };
static Fl_Input * modelPathPtr   { nullptr };
static Fl_Button* modelChoosePtr { nullptr };
static Fl_Input * audioPathPtr   { nullptr };
static Fl_Button* audioChoosePtr { nullptr };
static Fl_Button* pcmPtr         { nullptr };
static Fl_Button* trainPtr       { nullptr };
static Fl_Button* generatePtr    { nullptr };
static Fl_Button* finetunePtr    { nullptr };
static Fl_Button* quantizationPtr{ nullptr };
static Fl_Button* modelReleasePtr{ nullptr };

static std::mutex mutex;

static std::unique_ptr<lifuren::ComposeModelClient> composeClient{ nullptr };

static void pcmCallback            (Fl_Widget*, void*);
static void trainCallback          (Fl_Widget*, void*);
static void generateCallback       (Fl_Widget*, void*);
static void modelReleaseCallback   (Fl_Widget*, void*);
static void clientCallback         (Fl_Widget*, void*);
static void chooseFileCallback     (Fl_Widget*, void*);
static void chooseDirectoryCallback(Fl_Widget*, void*);

lifuren::AudioWindow::AudioWindow(int width, int height, const char* title) : Window(width, height, title) {
}

lifuren::AudioWindow::~AudioWindow() {
    // 保存配置
    this->saveConfig();
    // 释放资源
    LFR_DELETE_PTR(clientPtr);
    LFR_DELETE_PTR(pathPathPtr);
    LFR_DELETE_PTR(pathChoosePtr);
    LFR_DELETE_PTR(modelPathPtr);
    LFR_DELETE_PTR(modelChoosePtr);
    LFR_DELETE_PTR(audioPathPtr);
    LFR_DELETE_PTR(audioChoosePtr);
    LFR_DELETE_PTR(pcmPtr);
    LFR_DELETE_PTR(trainPtr);
    LFR_DELETE_PTR(generatePtr);
    LFR_DELETE_PTR(finetunePtr);
    LFR_DELETE_PTR(quantizationPtr);
    LFR_DELETE_PTR(modelReleasePtr);
}

void lifuren::AudioWindow::saveConfig() {
    lifuren::Configuration::saveConfig();
}

void lifuren::AudioWindow::redrawConfigElement() {
    const auto& audioConfig = lifuren::config::CONFIG.audio;
    pathPathPtr->value(audioConfig.path.c_str());
}

void lifuren::AudioWindow::drawElement() {
    // 绘制界面
    clientPtr       = new Fl_Choice( 80, 10,  200, 30, "终端名称");
    pathPathPtr     = new Fl_Input(  80, 50,  400, 30, "数据集路径");
    pathChoosePtr   = new Fl_Button(480, 50,  100, 30, "选择数据集");
    modelPathPtr    = new Fl_Input(  80, 90,  400, 30, "模型路径");
    modelChoosePtr  = new Fl_Button(480, 90,  100, 30, "选择模型");
    audioPathPtr    = new Fl_Input(  80, 130, 400, 30, "音频路径");
    audioChoosePtr  = new Fl_Button(480, 130, 100, 30, "选择音频");
    pcmPtr          = new Fl_Button( 80, 170, 100, 30, "PCM转换");
    trainPtr        = new Fl_Button(180, 170, 100, 30, "训练模型");
    generatePtr     = new Fl_Button(280, 170, 100, 30, "生成音频");
    finetunePtr     = new Fl_Button(380, 170, 100, 30, "模型微调");
    quantizationPtr = new Fl_Button(480, 170, 100, 30, "模型量化");
    modelReleasePtr = new Fl_Button(580, 170, 100, 30, "释放模型");
    // 绑定事件
    const auto& audioConfig = lifuren::config::CONFIG.audio;
    lifuren::fillChoice(clientPtr, audioConfig.clients, audioConfig.client);
    clientPtr->callback(clientCallback, this);
    pathChoosePtr->callback(chooseDirectoryCallback, pathPathPtr);
    modelChoosePtr->callback(chooseFileCallback, modelPathPtr);
    audioChoosePtr->callback(chooseFileCallback, audioPathPtr);
    pcmPtr->callback(pcmCallback, this);
    trainPtr->callback(trainCallback, this);
    generatePtr->callback(generateCallback, this);
    modelReleasePtr->callback(modelReleaseCallback, this);
    this->redrawConfigElement();
}

static void pcmCallback(Fl_Widget*, void*) {
}

static void trainCallback(Fl_Widget*, void*) {
}

static void generateCallback(Fl_Widget*, void*) {
    if(clientPtr->value() < 0) {
        fl_message("没有选择绘画终端");
        return;
    }
    {
        // TODO: 模型切换是否自动释放模型
        composeClient = lifuren::getComposeClient(clientPtr->text());
        if(!composeClient) {
            fl_message("不支持的终端");
            return;
        }
    }
}

static void modelReleaseCallback(Fl_Widget*, void*) {
    if(!composeClient) {
        return;
    }
    composeClient = nullptr;
}

static void clientCallback(Fl_Widget*, void* voidPtr) {
    lifuren::AudioWindow* windowPtr = static_cast<lifuren::AudioWindow*>(voidPtr);
    auto& audioConfig  = lifuren::config::CONFIG.audio;
    audioConfig.client = clientPtr->text();
    windowPtr->redrawConfigElement();
}

static void chooseFileCallback(Fl_Widget*, void* voidPtr) {
    std::string filename = lifuren::fileChooser("选择文件", "*.{aac,ogg,mp3,pt}");
    if(filename.empty()) {
        return;
    }
    Fl_Input* inputPtr = static_cast<Fl_Input*>(voidPtr);
    inputPtr->value(filename.c_str());
}

static void chooseDirectoryCallback(Fl_Widget*, void* voidPtr) {
    std::string filename = lifuren::directoryChooser("选择目录");
    if(filename.empty()) {
        return;
    }
    Fl_Input* inputPtr = static_cast<Fl_Input*>(voidPtr);
    inputPtr->value(filename.c_str());
}

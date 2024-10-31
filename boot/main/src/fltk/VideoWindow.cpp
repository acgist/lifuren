#include "lifuren/FLTK.hpp"

#include <mutex>
#include <thread>

#include "spdlog/spdlog.h"

#include "FL/fl_ask.H"
#include "FL/Fl_Input.H"
#include "FL/Fl_Button.H"
#include "FL/Fl_Choice.H"
#include "FL/Fl_Text_Buffer.H"
#include "FL/Fl_Text_Editor.H"

#include "lifuren/Raii.hpp"
#include "lifuren/Config.hpp"
#include "lifuren/ActClient.hpp"

static Fl_Choice* clientPtr      { nullptr };
static Fl_Input * modelPathPtr   { nullptr };
static Fl_Button* modelChoosePtr { nullptr };
static Fl_Input * videoPathPtr   { nullptr };
static Fl_Button* videoChoosePtr { nullptr };
static Fl_Input * outputPathPtr  { nullptr };
static Fl_Button* outputChoosePtr{ nullptr };
static Fl_Button* generatePtr    { nullptr };
static Fl_Button* trainPtr       { nullptr };
static Fl_Button* finetunePtr    { nullptr };
static Fl_Button* quantizationPtr{ nullptr };
static Fl_Button* modelReleasePtr{ nullptr };
static Fl_Text_Buffer* promptBufferPtr{ nullptr };
static Fl_Text_Editor* promptEditorPtr{ nullptr };

static std::mutex mutex;

static std::unique_ptr<lifuren::ActClient> actClient{ nullptr };

static void generate(Fl_Widget*, void*);
static void modelReleaseCallback(Fl_Widget*, void*);
static void clientCallback(Fl_Widget*, void*);
static void chooseFileCallback(Fl_Widget*, void*);
static void chooseDirectoryCallback(Fl_Widget*, void*);

lifuren::VideoWindow::VideoWindow(int width, int height, const char* title) : Window(width, height, title) {
}

lifuren::VideoWindow::~VideoWindow() {
    // 保存配置
    this->saveConfig();
    // 释放资源
    LFR_DELETE_PTR(clientPtr);
    LFR_DELETE_PTR(modelPathPtr);
    LFR_DELETE_PTR(modelChoosePtr);
    LFR_DELETE_PTR(videoPathPtr);
    LFR_DELETE_PTR(videoChoosePtr);
    LFR_DELETE_PTR(outputPathPtr);
    LFR_DELETE_PTR(outputChoosePtr);
    LFR_DELETE_PTR(generatePtr);
    LFR_DELETE_PTR(trainPtr);
    LFR_DELETE_PTR(finetunePtr);
    LFR_DELETE_PTR(quantizationPtr);
    LFR_DELETE_PTR(modelReleasePtr);
    LFR_DELETE_PTR(promptEditorPtr);
    LFR_DELETE_PTR(promptBufferPtr);
}

void lifuren::VideoWindow::saveConfig() {
    auto& videoConfig = lifuren::config::CONFIG.video;
    videoConfig.output = outputPathPtr->value();
    if(videoConfig.client == lifuren::config::CONFIG_ACT_GUANHANQIN) {
        auto& actGuanhanqin = lifuren::config::CONFIG.actGuanhanqin;
        actGuanhanqin.model = modelPathPtr->value();
    } else if(videoConfig.client == lifuren::config::CONFIG_ACT_TANGXIANZU) {
        auto& actTangxianzu = lifuren::config::CONFIG.actTangxianzu;
        actTangxianzu.model = modelPathPtr->value();
    } else {
        SPDLOG_WARN("不支持的绘画终端：{}", videoConfig.client);
    }
    lifuren::Configuration::saveConfig();
}

void lifuren::VideoWindow::redrawConfigElement() {
    const auto& videoConfig = lifuren::config::CONFIG.video;
    outputPathPtr->value(videoConfig.output.c_str());
    if(videoConfig.client == lifuren::config::CONFIG_ACT_GUANHANQIN) {
        const auto& actGuanhanqin = lifuren::config::CONFIG.actGuanhanqin;
        modelPathPtr->value(actGuanhanqin.model.c_str());
    } else if(videoConfig.client == lifuren::config::CONFIG_ACT_TANGXIANZU) {
        const auto& actTangxianzu = lifuren::config::CONFIG.actTangxianzu;
        modelPathPtr->value(actTangxianzu.model.c_str());
    } else {
        modelPathPtr->value("");
    }
}

void lifuren::VideoWindow::drawElement() {
    // 绘制界面
    promptEditorPtr = new Fl_Text_Editor(10, 20, this->w() - 20, 100, "提示内容");
    promptBufferPtr = new Fl_Text_Buffer();
    promptEditorPtr->begin();
    promptEditorPtr->buffer(promptBufferPtr);
    promptEditorPtr->wrap_mode(promptEditorPtr->WRAP_AT_COLUMN, promptEditorPtr->textfont());
    promptEditorPtr->end();
    videoPathPtr    = new Fl_Input( 70,  130, 400, 30, "视频路径");
    videoChoosePtr  = new Fl_Button(470, 130, 100, 30, "选择视频");
    clientPtr       = new Fl_Choice(70,  170, 200, 30, "终端名称");
    modelPathPtr    = new Fl_Input( 70,  210, 400, 30, "模型路径");
    modelChoosePtr  = new Fl_Button(470, 210, 100, 30, "选择模型");
    outputPathPtr   = new Fl_Input( 70,  250, 400, 30, "输出路径");
    outputChoosePtr = new Fl_Button(470, 250, 100, 30, "选择输出");
    generatePtr     = new Fl_Button(70,  290, 100, 30, "生成视频");
    trainPtr        = new Fl_Button(170, 290, 100, 30, "训练模型");
    finetunePtr     = new Fl_Button(270, 290, 100, 30, "模型微调");
    quantizationPtr = new Fl_Button(370, 290, 100, 30, "模型量化");
    modelReleasePtr = new Fl_Button(470, 290, 100, 30, "释放模型");
    // 绑定事件
    // 终端名称
    const auto& videoConfig = lifuren::config::CONFIG.video;
    lifuren::fillChoice(clientPtr, videoConfig.clients, videoConfig.client);
    clientPtr->callback(clientCallback, this);
    // 选择视频
    videoChoosePtr->callback(chooseFileCallback, videoPathPtr);
    // 选择模型
    modelChoosePtr->callback(chooseFileCallback, modelPathPtr);
    // 选择输出
    outputChoosePtr->callback(chooseDirectoryCallback, outputPathPtr);
    // 生成视频
    generatePtr->callback(generate, this);
    // 释放模型
    modelReleasePtr->callback(modelReleaseCallback, this);
    // 重绘配置
    this->redrawConfigElement();
}

static void generate(Fl_Widget*, void*) {
    if(clientPtr->value() < 0) {
        fl_message("没有选择绘画终端");
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mutex);
        if(actClient && actClient->isRunning()) {
            fl_message("上次绘画任务没有完成");
            return;
        }
        // TODO: 模型切换是否自动释放模型
        actClient = lifuren::ActClient::getClient(clientPtr->text());
        if(!actClient) {
            fl_message("不支持的终端");
            return;
        }
    }
    std::thread thread([]() {
    });
    thread.detach();
}

static void modelReleaseCallback(Fl_Widget*, void*) {
    if(!actClient) {
        return;
    }
    if(actClient->isRunning()) {
        fl_message("当前正在进行生成视频任务");
        return;
    }
    actClient = nullptr;
}

static void clientCallback(Fl_Widget*, void* voidPtr) {
    lifuren::VideoWindow* windowPtr = static_cast<lifuren::VideoWindow*>(voidPtr);
    auto& videoConfig  = lifuren::config::CONFIG.video;
    videoConfig.client = clientPtr->text();
    windowPtr->redrawConfigElement();
}

static void chooseFileCallback(Fl_Widget*, void* voidPtr) {
    std::string filename = lifuren::fileChooser("选择文件", "*.{mp4,pt}");
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

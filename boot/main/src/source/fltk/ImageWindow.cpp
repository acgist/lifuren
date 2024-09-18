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
#include "lifuren/PaintClient.hpp"

static Fl_Choice* clientPtr      { nullptr };
static Fl_Input * modelPathPtr   { nullptr };
static Fl_Button* modelChoosePtr { nullptr };
static Fl_Input * imagePathPtr   { nullptr };
static Fl_Button* imageChoosePtr { nullptr };
static Fl_Input * outputPathPtr  { nullptr };
static Fl_Button* outputChoosePtr{ nullptr };
static Fl_Button* generatePtr    { nullptr };
static Fl_Input * poetryPromptPtr{ nullptr };
static Fl_Button* poetrySearchPtr{ nullptr };
static Fl_Text_Buffer* promptBufferPtr{ nullptr };
static Fl_Text_Editor* promptEditorPtr{ nullptr };

static std::mutex mutex;

static std::unique_ptr<lifuren::PaintClient> paintClient{ nullptr };

static void generate(Fl_Widget*, void*);
static void clientCallback(Fl_Widget*, void*);
static void chooseFileCallback(Fl_Widget*, void*);
static void chooseDirectoryCallback(Fl_Widget*, void*);

lifuren::ImageWindow::ImageWindow(int width, int height, const char* title) : Window(width, height, title) {
}

lifuren::ImageWindow::~ImageWindow() {
    // 保存配置
    this->saveConfig();
    // 释放资源
    LFR_DELETE_PTR(clientPtr);
    LFR_DELETE_PTR(modelPathPtr);
    LFR_DELETE_PTR(modelChoosePtr);
    LFR_DELETE_PTR(imagePathPtr);
    LFR_DELETE_PTR(imageChoosePtr);
    LFR_DELETE_PTR(outputPathPtr);
    LFR_DELETE_PTR(outputChoosePtr);
    LFR_DELETE_PTR(generatePtr);
    LFR_DELETE_PTR(poetryPromptPtr);
    LFR_DELETE_PTR(poetrySearchPtr);
    LFR_DELETE_PTR(promptEditorPtr);
    LFR_DELETE_PTR(promptBufferPtr);
}

void lifuren::ImageWindow::saveConfig() {
    auto& imageConfig = lifuren::config::CONFIG.image;
    imageConfig.output = outputPathPtr->value();
    if(imageConfig.client == lifuren::config::CONFIG_STABLE_DIFFUSION_CPP) {
        auto& stableDiffusionCPP = lifuren::config::CONFIG.stableDiffusionCPP;
        stableDiffusionCPP.model = modelPathPtr->value();
    } else if(imageConfig.client == lifuren::config::CONFIG_PAINT_CYCLE_GAN) {
        auto& paintCycleGAN = lifuren::config::CONFIG.paintCycleGAN;
        paintCycleGAN.model = modelPathPtr->value();
    } else if(imageConfig.client == lifuren::config::CONFIG_PAINT_STYLE_GAN) {
        auto& paintSytleGAN = lifuren::config::CONFIG.paintSytleGAN;
        paintSytleGAN.model = modelPathPtr->value();
    } else {
        SPDLOG_WARN("不支持的绘画终端：{}", imageConfig.client);
    }
    lifuren::Configuration::saveConfig();
}

void lifuren::ImageWindow::redrawConfigElement() {
    const auto& imageConfig = lifuren::config::CONFIG.image;
    outputPathPtr->value(imageConfig.output.c_str());
    if(imageConfig.client == lifuren::config::CONFIG_STABLE_DIFFUSION_CPP) {
        const auto& stableDiffusionCPP = lifuren::config::CONFIG.stableDiffusionCPP;
        modelPathPtr->value(stableDiffusionCPP.model.c_str());
    } else if(imageConfig.client == lifuren::config::CONFIG_PAINT_CYCLE_GAN) {
        const auto& paintCycleGAN = lifuren::config::CONFIG.paintCycleGAN;
        modelPathPtr->value(paintCycleGAN.model.c_str());
    } else if(imageConfig.client == lifuren::config::CONFIG_PAINT_STYLE_GAN) {
        const auto& paintSytleGAN = lifuren::config::CONFIG.paintSytleGAN;
        modelPathPtr->value(paintSytleGAN.model.c_str());
    } else {
        modelPathPtr->value("");
    }
}

void lifuren::ImageWindow::drawElement() {
    // 绘制界面
    poetryPromptPtr = new Fl_Input( 10,  10, 300, 30);
    poetrySearchPtr = new Fl_Button(310, 10, 100, 30, "搜索诗词");
    promptEditorPtr = new Fl_Text_Editor(10, 50, this->w() - 20, 100, "提示内容");
    promptBufferPtr = new Fl_Text_Buffer();
    promptEditorPtr->begin();
    promptEditorPtr->buffer(promptBufferPtr);
    promptEditorPtr->wrap_mode(promptEditorPtr->WRAP_AT_COLUMN, promptEditorPtr->textfont());
    promptEditorPtr->end();
    imagePathPtr    = new Fl_Input( 70,  160, 400, 30, "图片路径");
    imageChoosePtr  = new Fl_Button(470, 160, 100, 30, "选择图片");
    clientPtr       = new Fl_Choice(70,  200, 200, 30, "终端名称");
    modelPathPtr    = new Fl_Input( 70,  240, 400, 30, "模型路径");
    modelChoosePtr  = new Fl_Button(470, 240, 100, 30, "选择模型");
    outputPathPtr   = new Fl_Input( 70,  280, 400, 30, "输出路径");
    outputChoosePtr = new Fl_Button(470, 280, 100, 30, "选择输出");
    generatePtr     = new Fl_Button(70,  320, 100, 30, "生成图片");
    // 绑定事件
    // 终端名称
    const auto& imageConfig = lifuren::config::CONFIG.image;
    lifuren::fillChoice(clientPtr, imageConfig.clients, imageConfig.client);
    clientPtr->callback(clientCallback, this);
    // 选择图片
    imageChoosePtr->callback(chooseFileCallback, imagePathPtr);
    // 选择模型
    modelChoosePtr->callback(chooseFileCallback, modelPathPtr);
    // 选择输出
    outputChoosePtr->callback(chooseDirectoryCallback, outputPathPtr);
    // 生成图片
    generatePtr->callback(generate, this);
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
        if(paintClient && paintClient->isRunning()) {
            fl_message("上次绘画任务没有完成");
            return;
        }
        paintClient = lifuren::PaintClient::getClient(clientPtr->text());
        if(!paintClient) {
            fl_message("不支持的终端");
            return;
        }
    }
    std::thread thread([]() {
        lifuren::PaintClient::PaintOptions options;
        options.mode   = std::strlen(imagePathPtr->value()) == 0LL ? lifuren::PaintClient::Mode::TXT2IMG : lifuren::PaintClient::Mode::IMG2IMG;
        options.image  = imagePathPtr->value();
        options.prompt = promptBufferPtr->text();
        #if defined(_DEBUG) || !defined(NDEBUG)
        options.steps  = 1;
        #endif
        paintClient->paint(options, [](bool finish, float percent, const std::string& message) {
            if(finish) {
                fl_message("绘制完成");
            } else {
                // 进度
            }
            return true;
        });
    });
    thread.detach();
}

static void chooseFileCallback(Fl_Widget*, void* voidPtr) {
    std::string filename = lifuren::fileChooser("选择文件");
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

static void clientCallback(Fl_Widget*, void* voidPtr) {
    lifuren::ImageWindow* windowPtr = static_cast<lifuren::ImageWindow*>(voidPtr);
    auto& imageConfig  = lifuren::config::CONFIG.image;
    imageConfig.client = clientPtr->text();
    windowPtr->redrawConfigElement();
}

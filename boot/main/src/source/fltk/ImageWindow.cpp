#include "lifuren/FLTK.hpp"

#include <algorithm>

#include "spdlog/spdlog.h"

#include "FL/Fl_Input.H"
#include "FL/Fl_Button.H"
#include "FL/Fl_Choice.H"
#include "FL/Fl_Text_Buffer.H"
#include "FL/Fl_Text_Editor.H"

static Fl_Choice* clientPtr       { nullptr };
static Fl_Input*  modelPathPtr    { nullptr };
static Fl_Button* modelChoosePtr  { nullptr };
static Fl_Input*  imagePathPtr    { nullptr };
static Fl_Button* imageChoosePtr  { nullptr };
static Fl_Input*  outputPathPtr   { nullptr };
static Fl_Button* outputChoosePtr { nullptr };
static Fl_Button* generatePtr     { nullptr };
static Fl_Input*  poetryPromptPtr { nullptr };
static Fl_Button* poetrySearchPtr { nullptr };
static Fl_Text_Buffer* promptBufferPtr{ nullptr };
static Fl_Text_Editor* promptEditorPtr{ nullptr };

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
    if(imageConfig.client == "stable-diffusion-cpp") {
        auto& stableDiffusionCPP = lifuren::config::CONFIG.stableDiffusionCPP;
        stableDiffusionCPP.model = modelPathPtr->value();
    } else {
    }
    lifuren::Configuration::saveConfig();
}

void lifuren::ImageWindow::redrawConfigElement() {
    auto& imageConfig = lifuren::config::CONFIG.image;
    outputPathPtr->value(imageConfig.output.c_str());
    if(imageConfig.client == "stable-diffusion-cpp") {
        auto& stableDiffusionCPP = lifuren::config::CONFIG.stableDiffusionCPP;
        modelPathPtr->value(stableDiffusionCPP.model.c_str());
    } else {
    }
}

void lifuren::ImageWindow::drawElement() {
    // 绘制界面
    poetryPromptPtr = new Fl_Input( 10,  10, 300, 30);
    poetrySearchPtr = new Fl_Button(310, 10, 100, 30, "搜索诗词");
    promptEditorPtr = new Fl_Text_Editor(10, 50, this->w() - 20, 100, "提示内容");
    promptBufferPtr = new Fl_Text_Buffer();
    promptEditorPtr->buffer(promptBufferPtr);
    promptEditorPtr->wrap_mode(promptEditorPtr->WRAP_AT_COLUMN, promptEditorPtr->textfont());
    promptEditorPtr->end();
    imagePathPtr     = new Fl_Input( 70,  160, 400, 30, "图片路径");
    imageChoosePtr   = new Fl_Button(470, 160, 100, 30, "选择图片");
    clientPtr        = new Fl_Choice(70,  200, 200, 30, "终端名称");
    modelPathPtr     = new Fl_Input( 70,  240, 400, 30, "模型路径");
    modelChoosePtr   = new Fl_Button(470, 240, 100, 30, "选择模型");
    outputPathPtr    = new Fl_Input( 70,  280, 400, 30, "输出路径");
    outputChoosePtr  = new Fl_Button(470, 280, 100, 30, "选择输出");
    generatePtr      = new Fl_Button(70,  320, 100, 30, "生成图片");
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

static void generate(Fl_Widget*, void* voidPtr) {
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
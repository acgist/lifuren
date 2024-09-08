#include "lifuren/FLTK.hpp"

#include "spdlog/spdlog.h"

#include "FL/Fl_Input.H"
#include "FL/Fl_Button.H"
#include "FL/Fl_Choice.H"
#include "FL/Fl_Text_Buffer.H"
#include "FL/Fl_Text_Editor.H"

static Fl_Choice* clientPtr     { nullptr };
static Fl_Input*  imagePathPtr  { nullptr };
static Fl_Button* imageChoosePtr{ nullptr };
static Fl_Input*  modelPathPtr  { nullptr };
static Fl_Button* modelChoosePtr{ nullptr };
static Fl_Button* generatePtr   { nullptr };
static Fl_Text_Buffer* promptBufferPtr{ nullptr };
static Fl_Text_Editor* promptEditorPtr{ nullptr };

static void generate(Fl_Widget*, void*);
static void clientCallback(Fl_Widget*, void*);

lifuren::PoetryWindow::PoetryWindow(int width, int height, const char* title) :ModelWindow(width, height, title) {
}

lifuren::PoetryWindow::~PoetryWindow() {
    // 保存配置
    this->saveConfig();
    // 释放资源
    LFR_DELETE_PTR(clientPtr);
    LFR_DELETE_PTR(imagePathPtr);
    LFR_DELETE_PTR(imageChoosePtr);
    LFR_DELETE_PTR(modelPathPtr);
    LFR_DELETE_PTR(modelChoosePtr);
    LFR_DELETE_PTR(generatePtr);
    LFR_DELETE_PTR(promptEditorPtr);
    LFR_DELETE_PTR(promptBufferPtr);
}

void lifuren::PoetryWindow::saveConfig() {
    auto& poetryConfig = lifuren::config::CONFIG.poetry;
    if(poetryConfig.client == "poetize-rnn") {
        auto& poetizeRNN = lifuren::config::CONFIG.poetizeRNN;
        poetizeRNN.model = modelPathPtr->value();
    } else {
    }
    lifuren::Configuration::saveConfig();
}

void lifuren::PoetryWindow::redrawConfigElement() {
    auto& poetryConfig = lifuren::config::CONFIG.poetry;
    if(poetryConfig.client == "poetize-rnn") {
        auto& poetizeRNN = lifuren::config::CONFIG.poetizeRNN;
        modelPathPtr->value(poetizeRNN.model.c_str());
    } else {
    }
}

void lifuren::PoetryWindow::drawElement() {
    // 绘制界面
    promptEditorPtr = new Fl_Text_Editor(10, 20, this->w() - 20, 100, "提示内容");
    promptBufferPtr = new Fl_Text_Buffer();
    promptEditorPtr->buffer(promptBufferPtr);
    promptEditorPtr->wrap_mode(promptEditorPtr->WRAP_AT_COLUMN, promptEditorPtr->textfont());
    promptEditorPtr->end();
    imagePathPtr     = new Fl_Input( 70,  130, 400, 30, "图片路径");
    imageChoosePtr   = new Fl_Button(470, 130, 100, 30, "选择图片");
    clientPtr        = new Fl_Choice(70,  170, 200, 30, "终端名称");
    modelPathPtr     = new Fl_Input( 70,  210, 400, 30, "模型路径");
    modelChoosePtr   = new Fl_Button(470, 210, 100, 30, "选择模型");
    generatePtr      = new Fl_Button(70,  250, 100, 30, "生成诗词");
    // 绑定事件
    // 终端名称
    const auto& poetryConfig = lifuren::config::CONFIG.poetry;
    lifuren::fillChoice(clientPtr, poetryConfig.clients, poetryConfig.client);
    clientPtr->callback(clientCallback, this);
    // 重绘配置
    this->redrawConfigElement();
}

static void generate(Fl_Widget*, void* voidPtr) {
}

static void clientCallback(Fl_Widget*, void* voidPtr) {
    lifuren::PoetryWindow* windowPtr = static_cast<lifuren::PoetryWindow*>(voidPtr);
    auto& poetryConfig = lifuren::config::CONFIG.poetry;
    poetryConfig.client = clientPtr->text();
    windowPtr->redrawConfigElement();
}

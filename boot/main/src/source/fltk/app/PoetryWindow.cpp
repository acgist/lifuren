#include "lifuren/FLTK.hpp"

#include "spdlog/spdlog.h"

#include "FL/Fl_Input.H"
#include "FL/Fl_Button.H"
#include "FL/Fl_Text_Buffer.H"
#include "FL/Fl_Text_Editor.H"

static Fl_Input*  imagePathPtr  { nullptr };
static Fl_Button* imageChoosePtr{ nullptr };
static Fl_Input*  modelPathPtr  { nullptr };
static Fl_Button* modelChoosePtr{ nullptr };
static Fl_Button* generatePtr   { nullptr };
static Fl_Text_Buffer* promptBufferPtr{ nullptr };
static Fl_Text_Editor* promptEditorPtr{ nullptr };

lifuren::PoetryWindow::PoetryWindow(int width, int height, const char* title) :ModelWindow(width, height, title) {
}

lifuren::PoetryWindow::~PoetryWindow() {
    SPDLOG_DEBUG("关闭窗口：{}", __FILE__);
    // 保存配置
    this->saveConfig();
    // 释放资源
    LFR_DELETE_PTR(imagePathPtr);
    LFR_DELETE_PTR(imageChoosePtr);
    LFR_DELETE_PTR(modelPathPtr);
    LFR_DELETE_PTR(modelChoosePtr);
    LFR_DELETE_PTR(generatePtr);
    LFR_DELETE_PTR(promptEditorPtr);
    LFR_DELETE_PTR(promptBufferPtr);
}

void lifuren::PoetryWindow::saveConfig() {
    lifuren::Configuration::saveConfig();
}

void lifuren::PoetryWindow::redrawConfigElement() {
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
    modelPathPtr     = new Fl_Input( 70,  170, 400, 30, "模型路径");
    modelChoosePtr   = new Fl_Button(470, 170, 100, 30, "选择模型");
    generatePtr      = new Fl_Button(70,  210, 100, 30, "生成图片");
    // 绑定事件
}

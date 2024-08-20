#include "lifuren/FLTK.hpp"

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
static Fl_Input*  commandPathPtr  { nullptr };
static Fl_Button* commandChoosePtr{ nullptr };
static Fl_Button* generatePtr     { nullptr };
static Fl_Text_Buffer* promptBufferPtr{ nullptr };
static Fl_Text_Editor* promptEditorPtr{ nullptr };

static void generate(Fl_Widget*, void*);
static void chooseCallback(Fl_Widget*, void*);

lifuren::ImageWindow::ImageWindow(int width, int height, const char* title) : ModelWindow(width, height, title) {
}

lifuren::ImageWindow::~ImageWindow() {
    SPDLOG_DEBUG("关闭窗口：{}", __FILE__);
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
    LFR_DELETE_PTR(commandPathPtr);
    LFR_DELETE_PTR(commandChoosePtr);
    LFR_DELETE_PTR(generatePtr);
    LFR_DELETE_PTR(promptEditorPtr);
    LFR_DELETE_PTR(promptBufferPtr);
}

void lifuren::ImageWindow::saveConfig() {
}

void lifuren::ImageWindow::redrawConfigElement() {
}

void lifuren::ImageWindow::drawElement() {
    // 绘制界面
    promptEditorPtr = new Fl_Text_Editor(10, 20, this->w() - 20, 100, "提示内容");
    promptBufferPtr = new Fl_Text_Buffer();
    promptEditorPtr->buffer(promptBufferPtr);
    promptEditorPtr->wrap_mode(promptEditorPtr->WRAP_AT_COLUMN, promptEditorPtr->textfont());
    promptEditorPtr->end();
    imagePathPtr     = new Fl_Input( 110, 130, 400, 30, "图片路径");
    imageChoosePtr   = new Fl_Button(510, 130, 100, 30, "选择图片");
    clientPtr        = new Fl_Choice(110, 170, 200, 30, "终端名称");
    modelPathPtr     = new Fl_Input( 110, 210, 400, 30, "模型路径");
    modelChoosePtr   = new Fl_Button(510, 210, 100, 30, "选择模型");
    outputPathPtr    = new Fl_Input( 110, 250, 400, 30, "输出路径");
    outputChoosePtr  = new Fl_Button(510, 250, 100, 30, "选择输出");
    commandPathPtr   = new Fl_Input( 110, 290, 400, 30, "命令路径");
    commandChoosePtr = new Fl_Button(510, 290, 100, 30, "选择命令");
    generatePtr      = new Fl_Button(110, 330, 100, 30, "生成图片");
    // 绑定事件
    // 终端名称
    const auto& image = lifuren::config::CONFIG.image;
    // 选择图片
    imageChoosePtr->callback(chooseCallback, imagePathPtr);
    // 选择模型
    modelChoosePtr->callback(chooseCallback, modelPathPtr);
    // 选择输出
    outputChoosePtr->callback(chooseCallback, outputPathPtr);
    // 选择命令
    commandChoosePtr->callback(chooseCallback, commandPathPtr);
    // 生成图片
    generatePtr->callback(generate, this);
}

static void generate(Fl_Widget*, void* voidPtr) {
}

static void chooseCallback(Fl_Widget*, void* voidPtr) {
    std::string filename = lifuren::directoryChooser("选择目录");
    if(filename.empty()) {
        return;
    }
    Fl_Input* inputPtr = static_cast<Fl_Input*>(voidPtr);
    inputPtr->value(filename.c_str());
}

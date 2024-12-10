#include "lifuren/FLTK.hpp"

#include "spdlog/spdlog.h"

#include "FL/fl_ask.H"
#include "FL/Fl_Input.H"
#include "FL/Fl_Button.H"
#include "FL/Fl_Choice.H"

#include "lifuren/Raii.hpp"
#include "lifuren/Config.hpp"
#include "lifuren/image/PaintClient.hpp"

static Fl_Choice* clientPtr      { nullptr };
static Fl_Input * pathPathPtr    { nullptr };
static Fl_Button* pathChoosePtr  { nullptr };
static Fl_Input * modelPathPtr   { nullptr };
static Fl_Button* modelChoosePtr { nullptr };
static Fl_Input * imagePathPtr   { nullptr };
static Fl_Button* imageChoosePtr { nullptr };
static Fl_Button* trainPtr       { nullptr };
static Fl_Button* generatePtr    { nullptr };
static Fl_Button* modelReleasePtr{ nullptr };

static std::unique_ptr<lifuren::PaintModelClient> paintClient{ nullptr };

static void trainCallback          (Fl_Widget*, void*);
static void generateCallback       (Fl_Widget*, void*);
static void modelReleaseCallback   (Fl_Widget*, void*);
static void clientCallback         (Fl_Widget*, void*);
static void chooseFileCallback     (Fl_Widget*, void*);
static void chooseDirectoryCallback(Fl_Widget*, void*);

lifuren::ImageWindow::ImageWindow(int width, int height, const char* title) : Window(width, height, title) {
}

lifuren::ImageWindow::~ImageWindow() {
    // 保存配置
    this->saveConfig();
    // 释放资源
    LFR_DELETE_PTR(clientPtr);
    LFR_DELETE_PTR(pathPathPtr);
    LFR_DELETE_PTR(pathChoosePtr);
    LFR_DELETE_PTR(modelPathPtr);
    LFR_DELETE_PTR(modelChoosePtr);
    LFR_DELETE_PTR(imagePathPtr);
    LFR_DELETE_PTR(imageChoosePtr);
    LFR_DELETE_PTR(trainPtr);
    LFR_DELETE_PTR(generatePtr);
    LFR_DELETE_PTR(modelReleasePtr);
}

void lifuren::ImageWindow::saveConfig() {
    lifuren::Configuration::saveConfig();
}

void lifuren::ImageWindow::drawElement() {
    // 绘制界面
    clientPtr       = new Fl_Choice( 80, 10,  200, 30, "终端名称");
    pathPathPtr     = new Fl_Input(  80, 50,  400, 30, "数据集路径");
    pathChoosePtr   = new Fl_Button(480, 50,  100, 30, "选择数据集");
    modelPathPtr    = new Fl_Input(  80, 90,  400, 30, "模型路径");
    modelChoosePtr  = new Fl_Button(480, 90,  100, 30, "选择模型");
    imagePathPtr    = new Fl_Input(  80, 130, 400, 30, "图片路径");
    imageChoosePtr  = new Fl_Button(480, 130, 100, 30, "选择图片");
    trainPtr        = new Fl_Button( 80, 170, 100, 30, "训练模型");
    generatePtr     = new Fl_Button(180, 170, 100, 30, "生成图片");
    modelReleasePtr = new Fl_Button(280, 170, 100, 30, "释放模型");
    // 绑定事件
    clientPtr->callback(clientCallback, this);
    pathChoosePtr->callback(chooseDirectoryCallback, pathPathPtr);
    modelChoosePtr->callback(chooseFileCallback, modelPathPtr);
    imageChoosePtr->callback(chooseFileCallback, imagePathPtr);
    trainPtr->callback(trainCallback, this);
    generatePtr->callback(generateCallback, this);
    modelReleasePtr->callback(modelReleaseCallback, this);
    // 默认数据
    const auto& imageConfig = lifuren::config::CONFIG.image;
    lifuren::fillChoice(clientPtr, imageConfig.clients, imageConfig.client);
    pathPathPtr->value(imageConfig.path.c_str());
    modelPathPtr->value(imageConfig.model.c_str());
}

static void trainCallback(Fl_Widget*, void*) {
    if(!paintClient) {
        fl_message("没有终端实例");
        return;
    }
}

static void generateCallback(Fl_Widget*, void*) {
    if(!paintClient) {
        fl_message("没有终端实例");
        return;
    }
}

static void modelReleaseCallback(Fl_Widget*, void*) {
    if(!paintClient) {
        return;
    }
    if(lifuren::ThreadWindow::checkImageThread()) {
        fl_message("当前还有任务运行不能释放模型：请先停止任务");
        return;
    }
    paintClient = nullptr;
}

static void clientCallback(Fl_Widget*, void* voidPtr) {
    lifuren::ImageWindow* windowPtr = static_cast<lifuren::ImageWindow*>(voidPtr);
    auto& imageConfig  = lifuren::config::CONFIG.image;
    imageConfig.client = clientPtr->text();
    paintClient        = lifuren::getPaintClient(imageConfig.client);
}

static void chooseFileCallback(Fl_Widget* widget, void* voidPtr) {
    lifuren::fileChooser(widget, voidPtr, "选择文件", "*.{png,jpg,jpeg,pt}");
}

static void chooseDirectoryCallback(Fl_Widget* widget, void* voidPtr) {
    lifuren::directoryChooser(widget, voidPtr, "选择目录");
}

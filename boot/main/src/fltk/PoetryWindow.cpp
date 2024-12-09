#include "lifuren/FLTK.hpp"

#include "spdlog/spdlog.h"

#include "FL/fl_ask.H"
#include "FL/Fl_Input.H"
#include "FL/Fl_Button.H"
#include "FL/Fl_Choice.H"

#include "lifuren/Raii.hpp"
#include "lifuren/Config.hpp"
#include "lifuren/poetry/PoetizeClient.hpp"

static Fl_Choice* clientPtr      { nullptr };
static Fl_Input * pathPathPtr    { nullptr };
static Fl_Button* pathChoosePtr  { nullptr };
static Fl_Input * modelPathPtr   { nullptr };
static Fl_Button* modelChoosePtr { nullptr };
static Fl_Choice* rhythmPtr      { nullptr };
static Fl_Input * promptPtr      { nullptr };
static Fl_Button* pepperPtr      { nullptr };
static Fl_Button* embeddingPtr   { nullptr };
static Fl_Button* trainPtr       { nullptr };
static Fl_Button* generatePtr    { nullptr };
static Fl_Button* finetunePtr    { nullptr };
static Fl_Button* quantizationPtr{ nullptr };
static Fl_Button* modelReleasePtr{ nullptr };

static std::unique_ptr<lifuren::PoetizeModelClient> poetizeClient{ nullptr };

static void trainCallback          (Fl_Widget*, void*);
static void generateCallback       (Fl_Widget*, void*);
static void clientCallback         (Fl_Widget*, void*);
static void modelReleaseCallback   (Fl_Widget*, void*);
static void chooseFileCallback     (Fl_Widget*, void*);
static void chooseDirectoryCallback(Fl_Widget*, void*);

lifuren::PoetryWindow::PoetryWindow(int width, int height, const char* title) : Window(width, height, title) {
}

lifuren::PoetryWindow::~PoetryWindow() {
    // 保存配置
    this->saveConfig();
    // 释放资源
    LFR_DELETE_PTR(clientPtr);
    LFR_DELETE_PTR(pathPathPtr);
    LFR_DELETE_PTR(pathChoosePtr);
    LFR_DELETE_PTR(modelPathPtr);
    LFR_DELETE_PTR(modelChoosePtr);
    LFR_DELETE_PTR(rhythmPtr);
    LFR_DELETE_PTR(promptPtr);
    LFR_DELETE_PTR(pepperPtr);
    LFR_DELETE_PTR(embeddingPtr);
    LFR_DELETE_PTR(trainPtr);
    LFR_DELETE_PTR(generatePtr);
    LFR_DELETE_PTR(finetunePtr);
    LFR_DELETE_PTR(quantizationPtr);
    LFR_DELETE_PTR(modelReleasePtr);
}

void lifuren::PoetryWindow::saveConfig() {
    lifuren::Configuration::saveConfig();
}

void lifuren::PoetryWindow::redrawConfigElement() {
    const auto& poetryConfig = lifuren::config::CONFIG.poetry;
    pathPathPtr->value(poetryConfig.path.c_str());
}

void lifuren::PoetryWindow::drawElement() {
    // 绘制界面
    clientPtr       = new Fl_Choice(80,  10,  200, 30, "终端名称");
    pathPathPtr     = new Fl_Input( 80,  50,  400, 30, "数据集路径");
    pathChoosePtr   = new Fl_Button(480, 50,  100, 30, "选择数据集");
    modelPathPtr    = new Fl_Input( 80,  90,  400, 30, "模型路径");
    modelChoosePtr  = new Fl_Button(480, 90,  100, 30, "选择模型");
    rhythmPtr       = new Fl_Choice(80,  130, 200, 30, "诗词格律");
    promptPtr       = new Fl_Input( 80,  170, 400, 30, "提示内容");
    pepperPtr       = new Fl_Button(80,  210, 100, 30, "辣椒嵌入");
    embeddingPtr    = new Fl_Button(180, 210, 100, 30, "诗词嵌入");
    trainPtr        = new Fl_Button(280, 210, 100, 30, "训练模型");
    generatePtr     = new Fl_Button(380, 210, 100, 30, "生成诗词");
    finetunePtr     = new Fl_Button(480, 210, 100, 30, "模型微调");
    quantizationPtr = new Fl_Button(580, 210, 100, 30, "模型量化");
    modelReleasePtr = new Fl_Button(680, 210, 100, 30, "释放模型");
    // 绑定事件
    const auto& poetryConfig = lifuren::config::CONFIG.poetry;
    lifuren::fillChoice(clientPtr, poetryConfig.clients, poetryConfig.client);
    clientPtr->callback(clientCallback, this);
    pathChoosePtr->callback(chooseDirectoryCallback, pathPathPtr);
    modelChoosePtr->callback(chooseFileCallback, modelPathPtr);
    lifuren::fillChoice(rhythmPtr, std::move(lifuren::config::all_rhythm()), "");
    trainPtr->callback(trainCallback, this);
    generatePtr->callback(generateCallback, this);
    modelReleasePtr->callback(modelReleaseCallback, this);
    this->redrawConfigElement();
}

static void trainCallback(Fl_Widget*, void*) {

}

static void generateCallback(Fl_Widget*, void* voidPtr) {
    if(clientPtr->value() < 0) {
        fl_message("没有选择诗词终端");
        return;
    }
    // TODO: 验证是否正在运行
    poetizeClient = lifuren::getPoetizeClient(clientPtr->text());
    if(!poetizeClient) {
        fl_message("不支持的终端：{}", clientPtr->text());
        return;
    }
}

static void modelReleaseCallback(Fl_Widget*, void*) {
    if(!poetizeClient) {
        return;
    }
    // TODO: 验证是否正在运行
    poetizeClient = nullptr;
}

static void clientCallback(Fl_Widget*, void* voidPtr) {
    lifuren::PoetryWindow* windowPtr = static_cast<lifuren::PoetryWindow*>(voidPtr);
    auto& poetryConfig = lifuren::config::CONFIG.poetry;
    poetryConfig.client = clientPtr->text();
    windowPtr->redrawConfigElement();
}

static void chooseFileCallback(Fl_Widget* widget, void* voidPtr) {
    lifuren::fileChooser(widget, voidPtr, "选择文件", "*.{pt}");
}

static void chooseDirectoryCallback(Fl_Widget* widget, void* voidPtr) {
    lifuren::directoryChooser(widget, voidPtr, "选择目录");
}

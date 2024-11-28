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
static Fl_Button* finetunePtr    { nullptr };
static Fl_Button* quantizationPtr{ nullptr };
static Fl_Button* modelReleasePtr{ nullptr };

static std::mutex mutex;

static std::unique_ptr<lifuren::PaintModelClient> paintClient{ nullptr };

static void trainCallback(Fl_Widget*, void*);
static void generateCallback(Fl_Widget*, void*);
static void modelReleaseCallback(Fl_Widget*, void*);
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
    LFR_DELETE_PTR(pathPathPtr);
    LFR_DELETE_PTR(pathChoosePtr);
    LFR_DELETE_PTR(modelPathPtr);
    LFR_DELETE_PTR(modelChoosePtr);
    LFR_DELETE_PTR(imagePathPtr);
    LFR_DELETE_PTR(imageChoosePtr);
    LFR_DELETE_PTR(trainPtr);
    LFR_DELETE_PTR(generatePtr);
    LFR_DELETE_PTR(finetunePtr);
    LFR_DELETE_PTR(quantizationPtr);
    LFR_DELETE_PTR(modelReleasePtr);
}

void lifuren::ImageWindow::saveConfig() {
    lifuren::Configuration::saveConfig();
}

void lifuren::ImageWindow::redrawConfigElement() {
    const auto& imageConfig = lifuren::config::CONFIG.image;
    pathPathPtr->value(imageConfig.path.c_str());
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
    finetunePtr     = new Fl_Button(280, 170, 100, 30, "模型微调");
    quantizationPtr = new Fl_Button(380, 170, 100, 30, "模型量化");
    modelReleasePtr = new Fl_Button(480, 170, 100, 30, "释放模型");
    // 绑定事件
    // 终端名称
    const auto& imageConfig = lifuren::config::CONFIG.image;
    lifuren::fillChoice(clientPtr, imageConfig.clients, imageConfig.client);
    clientPtr->callback(clientCallback, this);
    // 选择数据集
    pathChoosePtr->callback(chooseDirectoryCallback, pathPathPtr);
    // 选择模型
    modelChoosePtr->callback(chooseFileCallback, modelPathPtr);
    // 选择图片
    imageChoosePtr->callback(chooseFileCallback, imagePathPtr);
    // 训练模型
    trainPtr->callback(trainCallback, this);
    // 生成图片
    generatePtr->callback(generateCallback, this);
    // 释放模型
    modelReleasePtr->callback(modelReleaseCallback, this);
    // 重绘配置
    this->redrawConfigElement();
}

static void trainCallback(Fl_Widget*, void*) {

}

static void generateCallback(Fl_Widget*, void*) {
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
        // TODO: 模型切换是否自动释放模型
        paintClient = lifuren::getPaintClient(clientPtr->text());
        if(!paintClient) {
            fl_message("不支持的终端");
            return;
        }
    }
    std::thread thread([]() {
        lifuren::PaintOptions options;
        options.image  = imagePathPtr->value();
        #if defined(_DEBUG) || !defined(NDEBUG)
        #endif
        paintClient->pred(options, [](bool finish, float percent, const std::string& message) {
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

static void modelReleaseCallback(Fl_Widget*, void*) {
    if(!paintClient) {
        return;
    }
    if(paintClient->isRunning()) {
        fl_message("当前正在进行生成图片任务");
        return;
    }
    paintClient = nullptr;
}

static void clientCallback(Fl_Widget*, void* voidPtr) {
    lifuren::ImageWindow* windowPtr = static_cast<lifuren::ImageWindow*>(voidPtr);
    auto& imageConfig  = lifuren::config::CONFIG.image;
    imageConfig.client = clientPtr->text();
    windowPtr->redrawConfigElement();
}

static void chooseFileCallback(Fl_Widget*, void* voidPtr) {
    std::string filename = lifuren::fileChooser("选择文件", "*.{png,jpg,jpeg,pt}");
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

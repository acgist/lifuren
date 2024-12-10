#include "lifuren/FLTK.hpp"

#include "spdlog/spdlog.h"

#include "FL/fl_ask.H"
#include "FL/Fl_Input.H"
#include "FL/Fl_Button.H"
#include "FL/Fl_Choice.H"

#include "lifuren/File.hpp"
#include "lifuren/Raii.hpp"
#include "lifuren/Config.hpp"
#include "lifuren/video/ActClient.hpp"

static Fl_Choice* clientPtr      { nullptr };
static Fl_Input * pathPathPtr    { nullptr };
static Fl_Button* pathChoosePtr  { nullptr };
static Fl_Input * modelPathPtr   { nullptr };
static Fl_Button* modelChoosePtr { nullptr };
static Fl_Input * videoPathPtr   { nullptr };
static Fl_Button* videoChoosePtr { nullptr };
static Fl_Button* trainPtr       { nullptr };
static Fl_Button* generatePtr    { nullptr };
static Fl_Button* modelReleasePtr{ nullptr };

static std::unique_ptr<lifuren::ActModelClient> actClient{ nullptr };

static void trainCallback          (Fl_Widget*, void*);
static void generateCallback       (Fl_Widget*, void*);
static void modelReleaseCallback   (Fl_Widget*, void*);
static void clientCallback         (Fl_Widget*, void*);
static void chooseFileCallback     (Fl_Widget*, void*);
static void chooseDirectoryCallback(Fl_Widget*, void*);

lifuren::VideoWindow::VideoWindow(int width, int height, const char* title) : Window(width, height, title) {
}

lifuren::VideoWindow::~VideoWindow() {
    // 保存配置
    this->saveConfig();
    // 释放资源
    LFR_DELETE_PTR(clientPtr);
    LFR_DELETE_PTR(pathPathPtr);
    LFR_DELETE_PTR(pathChoosePtr);
    LFR_DELETE_PTR(modelPathPtr);
    LFR_DELETE_PTR(modelChoosePtr);
    LFR_DELETE_PTR(videoPathPtr);
    LFR_DELETE_PTR(videoChoosePtr);
    LFR_DELETE_PTR(trainPtr);
    LFR_DELETE_PTR(generatePtr);
    LFR_DELETE_PTR(modelReleasePtr);
}

void lifuren::VideoWindow::saveConfig() {
    lifuren::Configuration::saveConfig();
}

void lifuren::VideoWindow::drawElement() {
    // 绘制界面
    clientPtr       = new Fl_Choice( 80,  10, 200, 30, "终端名称");
    pathPathPtr     = new Fl_Input(  80,  50, 400, 30, "数据集路径");
    pathChoosePtr   = new Fl_Button(480,  50, 100, 30, "选择数据集");
    modelPathPtr    = new Fl_Input(  80,  90, 400, 30, "模型路径");
    modelChoosePtr  = new Fl_Button(480,  90, 100, 30, "选择模型");
    videoPathPtr    = new Fl_Input(  80, 130, 400, 30, "视频路径");
    videoChoosePtr  = new Fl_Button(480, 130, 100, 30, "选择视频");
    trainPtr        = new Fl_Button( 80, 170, 100, 30, "训练模型");
    generatePtr     = new Fl_Button(180, 170, 100, 30, "生成视频");
    modelReleasePtr = new Fl_Button(280, 170, 100, 30, "释放模型");
    // 绑定事件
    clientPtr->callback(clientCallback, this);
    pathChoosePtr->callback(chooseDirectoryCallback, pathPathPtr);
    modelChoosePtr->callback(chooseFileCallback, modelPathPtr);
    videoChoosePtr->callback(chooseFileCallback, videoPathPtr);
    trainPtr->callback(trainCallback, this);
    generatePtr->callback(generateCallback, this);
    modelReleasePtr->callback(modelReleaseCallback, this);
    // 默认数据
    const auto& videoConfig = lifuren::config::CONFIG.video;
    lifuren::fillChoice(clientPtr, videoConfig.clients, videoConfig.client);
    pathPathPtr->value(videoConfig.path.c_str());
    modelPathPtr->value(videoConfig.model.c_str());
}

static void trainCallback(Fl_Widget*, void*) {
    if(!actClient) {
        fl_message("没有终端实例");
        return;
    }
    const std::string path = pathPathPtr->value();
    if(path.empty()) {
        fl_message("没有选择数据集路径");
        return;
    }
    const std::string model_name = clientPtr->text();
    lifuren::ThreadWindow::startThread(
        lifuren::message::Type::VIDEO_MODEL_TRAIN,
        "视频模型训练",
        [path, model_name]() {
            lifuren::config::ModelParams params {
                .check_path = lifuren::file::join({path, lifuren::config::LIFUREN_HIDDEN_FILE}).string(),
                .model_name = model_name,
                .train_path = lifuren::file::join({path, lifuren::config::DATASET_TRAIN}).string(),
                .val_path   = lifuren::file::join({path, lifuren::config::DATASET_VAL}).string(),
                .test_path  = lifuren::file::join({path, lifuren::config::DATASET_TEST}).string(),
            };
            actClient->trainValAndTest(params);
            actClient->save(lifuren::file::join({path, lifuren::config::LIFUREN_HIDDEN_FILE}).string(), model_name + ".pt");
        }
    );
}

static void generateCallback(Fl_Widget*, void*) {
    if(!actClient) {
        fl_message("没有终端实例");
        return;
    }
    const std::string model = modelPathPtr->value();
    if(model.empty()) {
        fl_message("没有选择模型路径");
        return;
    }
    const std::string video = videoPathPtr->value();
    if(video.empty()) {
        fl_message("没有选择视频输入文件");
        return;
    }
    const std::string output = video + ".output.mp4";
    lifuren::ThreadWindow::startThread(
        lifuren::message::Type::VIDEO_MODEL_PRED,
        "生成视频",
        [model, video, output]() {
            lifuren::ActParams params {
                .model  = model,
                .video  = video,
                .output = output
            };
            actClient->pred(params);
        }
    );
}

static void modelReleaseCallback(Fl_Widget*, void*) {
    if(!actClient) {
        return;
    }
    if(lifuren::ThreadWindow::checkVideoThread()) {
        fl_message("当前还有任务运行不能释放模型：请先停止任务");
        return;
    }
    actClient = nullptr;
}

static void clientCallback(Fl_Widget*, void* voidPtr) {
    lifuren::VideoWindow* windowPtr = static_cast<lifuren::VideoWindow*>(voidPtr);
    auto& videoConfig  = lifuren::config::CONFIG.video;
    videoConfig.client = clientPtr->text();
    actClient          = lifuren::getActClient(videoConfig.client);
}

static void chooseFileCallback(Fl_Widget* widget, void* voidPtr) {
    lifuren::fileChooser(widget, voidPtr, "选择文件", "*.{mp4,pt}");
    auto& videoConfig = lifuren::config::CONFIG.video;
    if(voidPtr == modelPathPtr) {
        videoConfig.model = modelPathPtr->value();
    }
}

static void chooseDirectoryCallback(Fl_Widget* widget, void* voidPtr) {
    lifuren::directoryChooser(widget, voidPtr, "选择目录");
    auto& videoConfig = lifuren::config::CONFIG.video;
    if(voidPtr == pathPathPtr) {
        videoConfig.path = pathPathPtr->value();
    }
}

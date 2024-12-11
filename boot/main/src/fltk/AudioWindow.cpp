#include "lifuren/FLTK.hpp"

#include "spdlog/spdlog.h"

#include "FL/fl_ask.H"
#include "FL/Fl_Input.H"
#include "FL/Fl_Button.H"
#include "FL/Fl_Choice.H"

#include "lifuren/File.hpp"
#include "lifuren/Raii.hpp"
#include "lifuren/Config.hpp"
#include "lifuren/Dataset.hpp"
#include "lifuren/audio/Audio.hpp"
#include "lifuren/audio/ComposeClient.hpp"

static Fl_Choice* clientPtr      { nullptr };
static Fl_Input * pathPathPtr    { nullptr };
static Fl_Button* pathChoosePtr  { nullptr };
static Fl_Input * modelPathPtr   { nullptr };
static Fl_Button* modelChoosePtr { nullptr };
static Fl_Input * audioPathPtr   { nullptr };
static Fl_Button* audioChoosePtr { nullptr };
static Fl_Button* pcmPtr         { nullptr };
static Fl_Button* trainPtr       { nullptr };
static Fl_Button* generatePtr    { nullptr };
static Fl_Button* modelReleasePtr{ nullptr };

static std::unique_ptr<lifuren::ComposeModelClient> composeClient{ nullptr };

static void pcmCallback            (Fl_Widget*, void*);
static void trainCallback          (Fl_Widget*, void*);
static void generateCallback       (Fl_Widget*, void*);
static void modelReleaseCallback   (Fl_Widget*, void*);
static void clientCallback         (Fl_Widget*, void*);
static void chooseFileCallback     (Fl_Widget*, void*);
static void chooseDirectoryCallback(Fl_Widget*, void*);

lifuren::AudioWindow::AudioWindow(int width, int height, const char* title) : Window(width, height, title) {
}

lifuren::AudioWindow::~AudioWindow() {
    // 保存配置
    this->saveConfig();
    // 释放资源
    LFR_DELETE_PTR(clientPtr);
    LFR_DELETE_PTR(pathPathPtr);
    LFR_DELETE_PTR(pathChoosePtr);
    LFR_DELETE_PTR(modelPathPtr);
    LFR_DELETE_PTR(modelChoosePtr);
    LFR_DELETE_PTR(audioPathPtr);
    LFR_DELETE_PTR(audioChoosePtr);
    LFR_DELETE_PTR(pcmPtr);
    LFR_DELETE_PTR(trainPtr);
    LFR_DELETE_PTR(generatePtr);
    LFR_DELETE_PTR(modelReleasePtr);
}

void lifuren::AudioWindow::saveConfig() {
    lifuren::Configuration::saveConfig();
}

void lifuren::AudioWindow::drawElement() {
    // 绘制界面
    clientPtr       = new Fl_Choice( 80, 10,  200, 30, "终端名称");
    pathPathPtr     = new Fl_Input(  80, 50,  400, 30, "数据集路径");
    pathChoosePtr   = new Fl_Button(480, 50,  100, 30, "选择数据集");
    modelPathPtr    = new Fl_Input(  80, 90,  400, 30, "模型路径");
    modelChoosePtr  = new Fl_Button(480, 90,  100, 30, "选择模型");
    audioPathPtr    = new Fl_Input(  80, 130, 400, 30, "音频路径");
    audioChoosePtr  = new Fl_Button(480, 130, 100, 30, "选择音频");
    pcmPtr          = new Fl_Button( 80, 170, 100, 30, "PCM转换");
    trainPtr        = new Fl_Button(180, 170, 100, 30, "训练模型");
    generatePtr     = new Fl_Button(280, 170, 100, 30, "生成音频");
    modelReleasePtr = new Fl_Button(380, 170, 100, 30, "释放模型");
    // 绑定事件
    clientPtr->callback(clientCallback, this);
    pathChoosePtr->callback(chooseDirectoryCallback, pathPathPtr);
    modelChoosePtr->callback(chooseFileCallback, modelPathPtr);
    audioChoosePtr->callback(chooseFileCallback, audioPathPtr);
    pcmPtr->callback(pcmCallback, this);
    trainPtr->callback(trainCallback, this);
    generatePtr->callback(generateCallback, this);
    modelReleasePtr->callback(modelReleaseCallback, this);
    // 默认数据
    const auto& audioConfig = lifuren::config::CONFIG.audio;
    lifuren::fillChoice(clientPtr, audioConfig.clients, audioConfig.client);
    pathPathPtr->value(audioConfig.path.c_str());
    modelPathPtr->value(audioConfig.model.c_str());
}

static void pcmCallback(Fl_Widget*, void*) {
    const char* path = pathPathPtr->value();
    if(std::strlen(path) == 0) {
        fl_message("请选择数据集路径");
        return;
    }
    lifuren::ThreadWindow::startThread(
        lifuren::message::Type::AUDIO_AUDIO_TO_PCM,
        "PCM转换",
        []() {
            auto allFileToPCM = std::bind(&lifuren::audio::allFileToPCM, std::placeholders::_1);
            if(lifuren::dataset::allDatasetPreprocessing(pathPathPtr->value(), allFileToPCM)) {
                lifuren::message::sendMessage("PCM转换成功");
            } else {
                lifuren::message::sendMessage("PCM转换失败");
            }
        }
    );
}

static void trainCallback(Fl_Widget*, void*) {
    if(!composeClient) {
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
        lifuren::message::Type::AUDIO_MODEL_TRAIN,
        "音频模型训练",
        [path, model_name]() {
            lifuren::config::ModelParams params {
                .check_path = lifuren::file::join({path, lifuren::config::LIFUREN_HIDDEN_FILE}).string(),
                .model_name = model_name,
                .train_path = lifuren::file::join({path, lifuren::config::DATASET_TRAIN}).string(),
                .val_path   = lifuren::file::join({path, lifuren::config::DATASET_VAL}).string(),
                .test_path  = lifuren::file::join({path, lifuren::config::DATASET_TEST}).string(),
            };
            composeClient->trainValAndTest(params);
            composeClient->save(lifuren::file::join({path, lifuren::config::LIFUREN_HIDDEN_FILE}).string(), model_name + ".pt");
        }
    );
}

static void generateCallback(Fl_Widget*, void*) {
    if(!composeClient) {
        fl_message("没有终端实例");
        return;
    }
    const std::string model = modelPathPtr->value();
    if(model.empty()) {
        fl_message("没有选择模型路径");
        return;
    }
    const std::string audio = audioPathPtr->value();
    if(audio.empty()) {
        fl_message("没有选择音频输入文件");
        return;
    }
    const std::string output = audio + ".output.pcm";
    lifuren::ThreadWindow::startThread(
        lifuren::message::Type::AUDIO_MODEL_PRED,
        "生成音频",
        [model, audio, output]() {
            lifuren::ComposeParams params {
                .model  = model,
                .audio  = audio,
                .output = output
            };
            composeClient->pred(params);
            lifuren::audio::toFile(output);
        }
    );
}

static void modelReleaseCallback(Fl_Widget*, void*) {
    if(!composeClient) {
        return;
    }
    if(lifuren::ThreadWindow::checkAudioThread()) {
        fl_message("当前还有任务运行不能释放模型：请先停止任务");
        return;
    }
    composeClient = nullptr;
}

static void clientCallback(Fl_Widget*, void* voidPtr) {
    if(composeClient) {
        fl_message("当前已有终端运行：请先释放模型");
    }
    lifuren::AudioWindow* windowPtr = static_cast<lifuren::AudioWindow*>(voidPtr);
    auto& audioConfig  = lifuren::config::CONFIG.audio;
    audioConfig.client = clientPtr->text();
    composeClient      = lifuren::getComposeClient(audioConfig.client);
}

static void chooseFileCallback(Fl_Widget* widget, void* voidPtr) {
    lifuren::fileChooser(widget, voidPtr, "选择文件", "*.{aac,ogg,mp3,pt}");
    auto& audioConfig = lifuren::config::CONFIG.audio;
    if(voidPtr == modelPathPtr) {
        audioConfig.model = modelPathPtr->value();
    }
}

static void chooseDirectoryCallback(Fl_Widget* widget, void* voidPtr) {
    lifuren::directoryChooser(widget, voidPtr, "选择目录");
    auto& audioConfig = lifuren::config::CONFIG.audio;
    if(voidPtr == pathPathPtr) {
        audioConfig.path = pathPathPtr->value();
    }
}

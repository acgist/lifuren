#include "lifuren/FLTK.hpp"

#include "spdlog/spdlog.h"

#include "FL/fl_ask.H"
#include "FL/Fl_Input.H"
#include "FL/Fl_Button.H"
#include "FL/Fl_Choice.H"

#include "lifuren/File.hpp"
#include "lifuren/Raii.hpp"
#include "lifuren/Config.hpp"
#include "lifuren/audio/Audio.hpp"

static Fl_Choice* clientPtr      { nullptr };
static Fl_Input * pathPathPtr    { nullptr };
static Fl_Button* pathChoosePtr  { nullptr };
static Fl_Input * modelPathPtr   { nullptr };
static Fl_Button* modelChoosePtr { nullptr };
static Fl_Input * audioPathPtr   { nullptr };
static Fl_Button* audioChoosePtr { nullptr };
static Fl_Button* embeddingPtr   { nullptr };
static Fl_Button* trainPtr       { nullptr };
static Fl_Button* generatePtr    { nullptr };
static Fl_Button* modelReleasePtr{ nullptr };

static std::unique_ptr<lifuren::audio::AudioModelClient> audioClient{ nullptr };

static void embeddingCallback      (Fl_Widget*, void*);
static void trainCallback          (Fl_Widget*, void*);
static void generateCallback       (Fl_Widget*, void*);
static void modelReleaseCallback   (Fl_Widget*, void*);
static void clientCallback         (Fl_Widget*, void*);
static bool loadModelClient        ();
static void pathPathCallback       (Fl_Widget*, void*);
static void modelPathCallback      (Fl_Widget*, void*);
static void chooseFileCallback     (Fl_Widget*, void*);
static void chooseDirectoryCallback(Fl_Widget*, void*);

lifuren::AudioWindow::AudioWindow(int width, int height, const char* title) : Window(width, height, title) {
}

lifuren::AudioWindow::~AudioWindow() {
    LFR_DELETE_PTR(clientPtr);
    LFR_DELETE_PTR(pathPathPtr);
    LFR_DELETE_PTR(pathChoosePtr);
    LFR_DELETE_PTR(modelPathPtr);
    LFR_DELETE_PTR(modelChoosePtr);
    LFR_DELETE_PTR(audioPathPtr);
    LFR_DELETE_PTR(audioChoosePtr);
    LFR_DELETE_PTR(embeddingPtr);
    LFR_DELETE_PTR(trainPtr);
    LFR_DELETE_PTR(generatePtr);
    LFR_DELETE_PTR(modelReleasePtr);
}

void lifuren::AudioWindow::drawElement() {
    clientPtr       = new Fl_Choice( 80, 10,  200, 30, "模型名称");
    pathPathPtr     = new Fl_Input ( 80, 50,  400, 30, "数据集路径");
    pathChoosePtr   = new Fl_Button(480, 50,  100, 30, "选择数据集");
    modelPathPtr    = new Fl_Input ( 80, 90,  400, 30, "模型路径");
    modelChoosePtr  = new Fl_Button(480, 90,  100, 30, "选择模型");
    audioPathPtr    = new Fl_Input ( 80, 130, 400, 30, "音频路径");
    audioChoosePtr  = new Fl_Button(480, 130, 100, 30, "选择音频");
    embeddingPtr    = new Fl_Button( 80, 170, 100, 30, "音频嵌入");
    trainPtr        = new Fl_Button(180, 170, 100, 30, "训练模型");
    generatePtr     = new Fl_Button(280, 170, 100, 30, "生成音频");
    modelReleasePtr = new Fl_Button(380, 170, 100, 30, "释放模型");
}

void lifuren::AudioWindow::bindEvent() {
    clientPtr->callback(clientCallback, this);
    pathPathPtr->callback(pathPathCallback, this);
    pathChoosePtr->callback(chooseDirectoryCallback, pathPathPtr);
    modelPathPtr->callback(modelPathCallback, this);
    modelChoosePtr->callback(chooseFileCallback, modelPathPtr);
    audioChoosePtr->callback(chooseFileCallback, audioPathPtr);
    embeddingPtr->callback(embeddingCallback, this);
    trainPtr->callback(trainCallback, this);
    generatePtr->callback(generateCallback, this);
    modelReleasePtr->callback(modelReleaseCallback, this);
}

void lifuren::AudioWindow::fillData() {
    const auto& audioConfig = lifuren::config::CONFIG.audio;
    lifuren::fillChoice(clientPtr, audioConfig.clients, audioConfig.client);
    pathPathPtr->value(audioConfig.path.c_str());
    modelPathPtr->value(audioConfig.model.c_str());
}

static void embeddingCallback(Fl_Widget*, void*) {
    const char* path = pathPathPtr->value();
    if(std::strlen(path) == 0) {
        fl_message("请选择数据集路径");
        return;
    }
    lifuren::ThreadWindow::startThread(
        lifuren::message::Type::AUDIO_EMBEDDING,
        "音频嵌入",
        [path]() {
            if(lifuren::audio::datasetPreprocessing(path)) {
                SPDLOG_INFO("音频嵌入成功");
            } else {
                SPDLOG_INFO("音频嵌入失败");
            }
        }
    );
}

static void trainCallback(Fl_Widget*, void*) {
    if(!audioClient && !loadModelClient()) {
        return;
    }
    const std::string path = pathPathPtr->value();
    if(path.empty()) {
        fl_message("请选择数据集路径");
        return;
    }
    const std::string model_name = clientPtr->text();
    lifuren::ThreadWindow::startThread(
        lifuren::message::Type::AUDIO_MODEL_TRAIN,
        "音频模型训练",
        [path, model_name]() {
            lifuren::config::ModelParams params {
                .model_name = model_name,
                .check_path = lifuren::file::join({path, lifuren::config::LIFUREN_HIDDEN_FILE}).string(),
                .train_path = lifuren::file::join({path, lifuren::config::DATASET_TRAIN}).string(),
                .val_path   = lifuren::file::join({path, lifuren::config::DATASET_VAL  }).string(),
                .test_path  = lifuren::file::join({path, lifuren::config::DATASET_TEST }).string(),
            };
            audioClient->trainValAndTest(params);
            audioClient->save(lifuren::file::join({path, lifuren::config::LIFUREN_HIDDEN_FILE}).string(), model_name + ".pt");
            SPDLOG_INFO("音频模型训练完成");
        }
    );
}

static void generateCallback(Fl_Widget*, void*) {
    if(!audioClient && !loadModelClient()) {
        return;
    }
    const std::string model = modelPathPtr->value();
    if(model.empty()) {
        fl_message("请选择模型路径");
        return;
    }
    const std::string audio = audioPathPtr->value();
    if(audio.empty()) {
        fl_message("请选择音频输入文件");
        return;
    }
    const std::string output = audio + ".output.pcm";
    lifuren::ThreadWindow::startThread(
        lifuren::message::Type::AUDIO_MODEL_PRED,
        "生成音频",
        [model, audio, output]() {
            lifuren::audio::AudioParams params {
                .model  = model,
                .audio  = audio,
                .output = output
            };
            const auto [success, output_file] = audioClient->pred(params);
            if(success) {
                SPDLOG_INFO("音频生成完成：{}", output_file);
            } else {
                SPDLOG_WARN("音频生成失败：{}", output_file);
            }
        }
    );
}

static void modelReleaseCallback(Fl_Widget*, void*) {
    if(!audioClient) {
        return;
    }
    if(lifuren::ThreadWindow::checkAudioThread()) {
        fl_message("当前还有任务运行不能释放模型");
        return;
    }
    audioClient = nullptr;
}

static void clientCallback(Fl_Widget*, void* voidPtr) {
    if(audioClient) {
        fl_message("请先释放模型");
        return;
    }
    lifuren::AudioWindow* windowPtr = static_cast<lifuren::AudioWindow*>(voidPtr);
    auto& audioConfig  = lifuren::config::CONFIG.audio;
    audioConfig.client = clientPtr->text();
    loadModelClient();
}

static bool loadModelClient() {
    audioClient = lifuren::audio::getAudioClient(clientPtr->text());
    if(!audioClient) {
        fl_message("不支持的模型终端");
        return false;
    }
    return true;
}

static void pathPathCallback(Fl_Widget*, void* voidPtr) {
    lifuren::config::CONFIG.audio.path = pathPathPtr->value();
}

static void modelPathCallback(Fl_Widget*, void* voidPtr) {
    lifuren::config::CONFIG.audio.model = modelPathPtr->value();
}

static void chooseFileCallback(Fl_Widget* widget, void* voidPtr) {
    lifuren::fileChooser(widget, voidPtr, "选择文件", "*.{aac,ogg,mp3,pt}");
    auto& audioConfig = lifuren::config::CONFIG.audio;
    if(voidPtr == modelPathPtr) {
        audioConfig.model = modelPathPtr->value();
    } else {
    }
}

static void chooseDirectoryCallback(Fl_Widget* widget, void* voidPtr) {
    lifuren::directoryChooser(widget, voidPtr, "选择目录");
    auto& audioConfig = lifuren::config::CONFIG.audio;
    if(voidPtr == pathPathPtr) {
        audioConfig.path = pathPathPtr->value();
    } else {
    }
}

#include "lifuren/FLTK.hpp"

#include "spdlog/spdlog.h"

#include "FL/fl_ask.H"
#include "FL/Fl_Input.H"
#include "FL/Fl_Button.H"
#include "FL/Fl_Choice.H"

#include "lifuren/File.hpp"
#include "lifuren/Raii.hpp"
#include "lifuren/Config.hpp"
#include "lifuren/String.hpp"
#include "lifuren/poetry/Poetry.hpp"

static Fl_Choice* clientPtr       { nullptr };
static Fl_Choice* ragTypePtr      { nullptr };
static Fl_Choice* embeddingTypePtr{ nullptr };
static Fl_Input * pathPathPtr     { nullptr };
static Fl_Button* pathChoosePtr   { nullptr };
static Fl_Input * modelPathPtr    { nullptr };
static Fl_Button* modelChoosePtr  { nullptr };
static Fl_Choice* rhythmPtr       { nullptr };
static Fl_Input * promptPtr       { nullptr };
static Fl_Button* pepperPtr       { nullptr };
static Fl_Button* embeddingPtr    { nullptr };
static Fl_Button* trainPtr        { nullptr };
static Fl_Button* generatePtr     { nullptr };
static Fl_Button* modelReleasePtr { nullptr };

static std::unique_ptr<lifuren::poetry::PoetryModelClient> poetryClient{ nullptr };

static void pepperCallback         (Fl_Widget*, void*);
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

lifuren::PoetryWindow::PoetryWindow(int width, int height, const char* title) : Window(width, height, title) {
}

lifuren::PoetryWindow::~PoetryWindow() {
    LFR_DELETE_PTR(clientPtr);
    LFR_DELETE_PTR(ragTypePtr);
    LFR_DELETE_PTR(embeddingTypePtr);
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
    LFR_DELETE_PTR(modelReleasePtr);
}

void lifuren::PoetryWindow::drawElement() {
    clientPtr        = new Fl_Choice( 80, 10,  200, 30, "模型名称");
    ragTypePtr       = new Fl_Choice( 80, 50,  200, 30, "向量库终端");
    embeddingTypePtr = new Fl_Choice( 80, 90,  200, 30, "嵌入终端");
    pathPathPtr      = new Fl_Input ( 80, 130, 400, 30, "数据集路径");
    pathChoosePtr    = new Fl_Button(480, 130, 100, 30, "选择数据集");
    modelPathPtr     = new Fl_Input ( 80, 170, 400, 30, "模型路径");
    modelChoosePtr   = new Fl_Button(480, 170, 100, 30, "选择模型");
    rhythmPtr        = new Fl_Choice( 80, 210, 200, 30, "诗词格律");
    promptPtr        = new Fl_Input ( 80, 250, 400, 30, "提示内容");
    pepperPtr        = new Fl_Button( 80, 290, 100, 30, "辣椒嵌入");
    embeddingPtr     = new Fl_Button(180, 290, 100, 30, "诗词嵌入");
    trainPtr         = new Fl_Button(280, 290, 100, 30, "训练模型");
    generatePtr      = new Fl_Button(380, 290, 100, 30, "生成诗词");
    modelReleasePtr  = new Fl_Button(480, 290, 100, 30, "释放模型");
}

void lifuren::PoetryWindow::bindEvent() {
    clientPtr->callback(clientCallback, this);
    pathPathPtr->callback(pathPathCallback, this);
    pathChoosePtr->callback(chooseDirectoryCallback, pathPathPtr);
    modelPathPtr->callback(modelPathCallback, this);
    modelChoosePtr->callback(chooseFileCallback, modelPathPtr);
    pepperPtr->callback(pepperCallback, this);
    embeddingPtr->callback(embeddingCallback, this);
    trainPtr->callback(trainCallback, this);
    generatePtr->callback(generateCallback, this);
    modelReleasePtr->callback(modelReleaseCallback, this);
}

void lifuren::PoetryWindow::fillData() {
    pepperPtr->tooltip("诗词分词去重通过ollama转为词嵌入文件加速诗词嵌入");
    const auto& poetryConfig = lifuren::config::CONFIG.poetry;
    lifuren::fillChoice(clientPtr, poetryConfig.clients, poetryConfig.client);
    lifuren::fillChoice(ragTypePtr, { lifuren::config::CONFIG_FAISS, lifuren::config::CONFIG_ELASTICSEARCH }, lifuren::config::CONFIG_FAISS);
    lifuren::fillChoice(embeddingTypePtr, { lifuren::config::CONFIG_PEPPER, lifuren::config::CONFIG_OLLAMA }, lifuren::config::CONFIG_PEPPER);
    lifuren::fillChoice(rhythmPtr, std::move(lifuren::config::all_rhythm()), "");
    pathPathPtr->value(poetryConfig.path.c_str());
    modelPathPtr->value(poetryConfig.model.c_str());
}

static void pepperCallback(Fl_Widget*, void*) {
    const std::string path = pathPathPtr->value();
    if(path.empty()) {
        fl_message("请选择数据集路径");
        return;
    }
    lifuren::ThreadWindow::startThread(
        lifuren::message::Type::POETRY_EMBEDDING_PEPPER,
        "辣椒嵌入",
        [path]() {
            if(lifuren::poetry::datasetPepperPreprocessing(path)) {
                SPDLOG_INFO("辣椒嵌入成功");
            } else {
                SPDLOG_WARN("辣椒嵌入失败");
            }
        }
    );
}

static void embeddingCallback(Fl_Widget*, void*) {
    const std::string path = pathPathPtr->value();
    if(path.empty()) {
        fl_message("请选择数据集路径");
        return;
    }
    lifuren::ThreadWindow::startThread(
        lifuren::message::Type::POETRY_EMBEDDING_POETRY,
        "诗词嵌入",
        [path]() {
            if(lifuren::poetry::datasetPoetryPreprocessing(path, ragTypePtr->text(), embeddingTypePtr->text())) {
                SPDLOG_INFO("诗词嵌入成功");
            } else {
                SPDLOG_WARN("诗词嵌入失败");
            }
        }
    );
}

static void trainCallback(Fl_Widget*, void*) {
    if(!poetryClient && !loadModelClient()) {
        return;
    }
    const std::string path = pathPathPtr->value();
    if(path.empty()) {
        fl_message("请选择数据集路径");
        return;
    }
    const std::string model_name = clientPtr->text();
    lifuren::ThreadWindow::startThread(
        lifuren::message::Type::POETRY_MODEL_TRAIN,
        "诗词模型训练",
        [path, model_name]() {
            lifuren::config::ModelParams params {
                .model_name = model_name,
                .check_path = lifuren::file::join({path, lifuren::config::LIFUREN_HIDDEN_FILE}).string(),
                .train_path = lifuren::file::join({path, lifuren::config::DATASET_TRAIN}).string(),
                .val_path   = lifuren::file::join({path, lifuren::config::DATASET_VAL  }).string(),
                .test_path  = lifuren::file::join({path, lifuren::config::DATASET_TEST }).string(),
            };
            poetryClient->trainValAndTest(params);
            poetryClient->save(lifuren::file::join({path, lifuren::config::LIFUREN_HIDDEN_FILE}).string(), model_name + ".pt");
            SPDLOG_INFO("诗词模型训练完成");
        }
    );
}

static void generateCallback(Fl_Widget*, void* voidPtr) {
    if(!poetryClient && !loadModelClient()) {
        return;
    }
    const std::string model = modelPathPtr->value();
    if(model.empty()) {
        fl_message("请选择模型路径");
        return;
    }
    if(rhythmPtr->value() <= 0) {
        fl_message("请选择格律");
        return;
    }
    const std::string rhythm = rhythmPtr->text();
    const std::string prompt = promptPtr->value();
    if(prompt.empty()) {
        fl_message("请输入提示内容");
        return;
    }
    std::vector<std::string> prompts = std::move(lifuren::string::split(prompt, " "));
    if(prompts.empty()) {
        fl_message("请输入提示内容");
        return;
    }
    lifuren::ThreadWindow::startThread(
        lifuren::message::Type::POETRY_MODEL_PRED,
        "生成诗词",
        [model, rhythm, prompts]() {
            lifuren::poetry::PoetryParams params {
                .model   = model,
                .rhythm  = rhythm,
                .prompts = prompts
            };
            const auto [success, result] = poetryClient->pred(params);
            if(success) {
                SPDLOG_INFO("诗词生成完成：{}", result);
            } else {
                SPDLOG_WARN("诗词生成失败：{}", result);
            }
        }
    );
}

static void modelReleaseCallback(Fl_Widget*, void*) {
    if(!poetryClient) {
        return;
    }
    if(lifuren::ThreadWindow::checkPoetryThread()) {
        fl_message("当前还有任务运行不能释放模型");
        return;
    }
    poetryClient = nullptr;
}

static void clientCallback(Fl_Widget*, void* voidPtr) {
    if(poetryClient) {
        fl_message("请先释放模型");
        return;
    }
    lifuren::PoetryWindow* windowPtr = static_cast<lifuren::PoetryWindow*>(voidPtr);
    auto& poetryConfig  = lifuren::config::CONFIG.poetry;
    poetryConfig.client = clientPtr->text();
    loadModelClient();
}

static bool loadModelClient() {
    poetryClient = lifuren::poetry::getPoetryClient(clientPtr->text());
    if(!poetryClient) {
        fl_message("不支持的模型终端");
        return false;
    }
    return true;
}

static void pathPathCallback(Fl_Widget*, void* voidPtr) {
    lifuren::config::CONFIG.poetry.path = pathPathPtr->value();
}

static void modelPathCallback(Fl_Widget*, void* voidPtr) {
    lifuren::config::CONFIG.poetry.model = modelPathPtr->value();
}

static void chooseFileCallback(Fl_Widget* widget, void* voidPtr) {
    lifuren::fileChooser(widget, voidPtr, "选择文件", "*.{pt}");
    auto& poetryConfig = lifuren::config::CONFIG.poetry;
    if(voidPtr == modelPathPtr) {
        poetryConfig.model = modelPathPtr->value();
    } else {
    }
}

static void chooseDirectoryCallback(Fl_Widget* widget, void* voidPtr) {
    lifuren::directoryChooser(widget, voidPtr, "选择目录");
    auto& poetryConfig = lifuren::config::CONFIG.poetry;
    if(voidPtr == pathPathPtr) {
        poetryConfig.path = pathPathPtr->value();
    } else {
    }
}

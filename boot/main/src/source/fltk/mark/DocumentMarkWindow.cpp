#include "lifuren/FLTK.hpp"

#include <algorithm>

#include "FL/Fl_Input.H"
#include "FL/Fl_Button.H"
#include "FL/Fl_Choice.H"

#include "spdlog/spdlog.h"

#include "lifuren/RAG.hpp"
#include "lifuren/Strings.hpp"

static Fl_Button* newPtr     { nullptr };
static Fl_Choice* pathPtr    { nullptr };
static Fl_Button* deletePtr  { nullptr };
static Fl_Choice* ragPtr     { nullptr };
static Fl_Button* markPtr    { nullptr };
static Fl_Button* stopPtr    { nullptr };
static Fl_Choice* chunkPtr   { nullptr };
static Fl_Input*  apiPtr     { nullptr };
static Fl_Input*  usernamePtr{ nullptr };
static Fl_Input*  passwordPtr{ nullptr };
static Fl_Choice* authTypePtr{ nullptr };
static Fl_Choice* embeddingPtr     { nullptr };
static Fl_Input*  embeddingPathPtr { nullptr };
static Fl_Input*  embeddingModelPtr{ nullptr };

static void newCallback   (Fl_Widget*, void*);
static void pathCallback  (Fl_Widget*, void*);
static void deleteCallback(Fl_Widget*, void*);
static void markCallback  (Fl_Widget*, void*);
static void stopCallback  (Fl_Widget*, void*);
static bool reloadConfig(lifuren::DocumentMarkWindow*, const std::string&);
static void percentCallback(float, bool);

static std::shared_ptr<lifuren::RAGTaskRunner> ragTaskRunner{ nullptr };
static lifuren::config::DocumentMarkConfig* documentMarkConfig{ nullptr };

lifuren::DocumentMarkWindow::DocumentMarkWindow(int width, int height, const char* title) : MarkWindow(width, height, title) {
}

lifuren::DocumentMarkWindow::~DocumentMarkWindow() {
    SPDLOG_DEBUG("关闭窗口：{}", __FILE__);
    // 保存配置
    this->saveConfig();
    // 清理数据
    if(ragTaskRunner) {
        ragTaskRunner->unregisterCallback();
        ragTaskRunner = nullptr;
    }
    documentMarkConfig = nullptr;
    // 释放资源
    LFR_DELETE_PTR(newPtr);
    LFR_DELETE_PTR(pathPtr);
    LFR_DELETE_PTR(deletePtr);
    LFR_DELETE_PTR(ragPtr);
    LFR_DELETE_PTR(markPtr);
    LFR_DELETE_PTR(stopPtr);
    LFR_DELETE_PTR(chunkPtr);
    LFR_DELETE_PTR(apiPtr);
    LFR_DELETE_PTR(usernamePtr);
    LFR_DELETE_PTR(passwordPtr);
    LFR_DELETE_PTR(authTypePtr);
    LFR_DELETE_PTR(embeddingPtr);
    LFR_DELETE_PTR(embeddingPathPtr);
    LFR_DELETE_PTR(embeddingModelPtr);
}

void lifuren::DocumentMarkWindow::saveConfig() {
    if(documentMarkConfig) {
        LFR_CHOICE_GET_DEFAULT(documentMarkConfig->rag, ragPtr);
        LFR_CHOICE_GET_DEFAULT(documentMarkConfig->chunk, chunkPtr);
        LFR_CHOICE_GET_DEFAULT(documentMarkConfig->embedding, embeddingPtr);
        if(documentMarkConfig->embedding == "ollama") {
            auto& ollamaConfig    = lifuren::config::CONFIG.ollama;
            ollamaConfig.api      = apiPtr->value();
            ollamaConfig.username = usernamePtr->value();
            ollamaConfig.password = passwordPtr->value();
            LFR_CHOICE_GET_DEFAULT(ollamaConfig.authType, authTypePtr);
            auto& embeddingClientConfig = ollamaConfig.embeddingClient;
            embeddingClientConfig.path  = embeddingPathPtr->value();
            embeddingClientConfig.model = embeddingModelPtr->value();
        } else {
        }
    }
    lifuren::Configuration::saveConfig();
}

void lifuren::DocumentMarkWindow::redrawConfigElement() {
    if(documentMarkConfig) {
        LFR_CHOICE_SET_DEFAULT(ragPtr,       documentMarkConfig->rag);
        LFR_CHOICE_SET_DEFAULT(chunkPtr,     documentMarkConfig->chunk);
        LFR_CHOICE_SET_DEFAULT(embeddingPtr, documentMarkConfig->embedding);
        if(documentMarkConfig->embedding == "ollama") {
            const auto& ollamaConfig = lifuren::config::CONFIG.ollama;
            apiPtr->value(ollamaConfig.api.c_str());
            usernamePtr->value(ollamaConfig.username.c_str());
            passwordPtr->value(ollamaConfig.password.c_str());
            LFR_CHOICE_SET_DEFAULT(authTypePtr, ollamaConfig.authType);
            const auto& embeddingClientConfig = ollamaConfig.embeddingClient;
            embeddingPathPtr->value(embeddingClientConfig.path.c_str());
            embeddingModelPtr->value(embeddingClientConfig.model.c_str());
        } else {
        }
    } else {

    }
}

void lifuren::DocumentMarkWindow::drawElement() {
    // 布局
    pathPtr           = new Fl_Choice(110, 10,  200,             30, "文档目录");
    newPtr            = new Fl_Button(310, 10,  100,             30, "新增目录");
    deletePtr         = new Fl_Button(410, 10,  100,             30, "删除目录");
    ragPtr            = new Fl_Choice(110, 50,  200,             30, "检索策略");
    chunkPtr          = new Fl_Choice(110, 90,  200,             30, "分段策略");
    embeddingPtr      = new Fl_Choice(110, 130, 200,             30, "词嵌入策略");
    apiPtr            = new Fl_Input(110,  170, this->w() - 200, 30, "服务地址");
    usernamePtr       = new Fl_Input(110,  210, this->w() - 200, 30, "账号");
    passwordPtr       = new Fl_Input(110,  250, this->w() - 200, 30, "密码");
    authTypePtr       = new Fl_Choice(110, 290, 200,             30, "授权类型"); 
    embeddingPathPtr  = new Fl_Input(110,  330, this->w() - 200, 30, "词嵌入地址");
    embeddingModelPtr = new Fl_Input(110,  370, this->w() - 200, 30, "词嵌入模型");
    markPtr           = new Fl_Button(110, 410, 100,             30, "开始标记");
    stopPtr           = new Fl_Button(210, 410, 100,             30, "停止标记");
    // 事件
    // 文档目录
    const auto& documentMark = lifuren::config::CONFIG.documentMark;
    for(auto& value : documentMark) {
        std::string path = value.path;
        LFR_CHOICE_TRANSFER(path);
        pathPtr->add(path.c_str());
    }
    pathPtr->callback(pathCallback, this);
    // 新增目录
    newPtr->callback(newCallback, this);
    // 删除目录
    deletePtr->callback(deleteCallback, this);
    // 检索策略
    ragPtr->add("Chroma");
    ragPtr->add("Typesense");
    ragPtr->add("ElasticSearch");
    // 分段策略
    chunkPtr->add("LINE");
    chunkPtr->add("TITLE");
    // 词嵌入策略
    embeddingPtr->add("ollama");
    embeddingPtr->add("Chinese-Word-Vectors");
    // 授权类型
    authTypePtr->add("NONE");
    authTypePtr->add("Basic");
    authTypePtr->add("Token");
    // 开始标记
    markPtr->callback(markCallback, this);
    // 停止标记
    stopPtr->callback(stopCallback, this);
}

static void newCallback(Fl_Widget*, void* voidPtr) {
    std::string filename = lifuren::directoryChooser("选择文档目录");
    if(filename.empty()) {
        return;
    }
    lifuren::DocumentMarkWindow* windowPtr = static_cast<lifuren::DocumentMarkWindow*>(voidPtr);
    if(reloadConfig(windowPtr, filename)) {
        std::string path = filename;
        LFR_CHOICE_TRANSFER(path);
        pathPtr->add(path.c_str());
    }
    pathPtr->value(pathPtr->find_index(filename.c_str()));
}

static void pathCallback(Fl_Widget*, void* voidPtr) {
    lifuren::DocumentMarkWindow* windowPtr = static_cast<lifuren::DocumentMarkWindow*>(voidPtr);
    windowPtr->saveConfig();
    reloadConfig(windowPtr, pathPtr->text());
}

static void deleteCallback(Fl_Widget*, void* voidPtr) {
    int index = pathPtr->value();
    if(index < 0) {
        return;
    }
    const std::string path = pathPtr->text();
    lifuren::DocumentMarkWindow* windowPtr = static_cast<lifuren::DocumentMarkWindow*>(voidPtr);
    auto& documentMarkConfig = lifuren::config::CONFIG.documentMark;
    auto iterator = std::find(documentMarkConfig.begin(), documentMarkConfig.end(), pathPtr->text());
    if(iterator != documentMarkConfig.end()) {
        documentMarkConfig.erase(iterator);
    }
    pathPtr->remove(index);
    ::documentMarkConfig = nullptr;
    if(ragTaskRunner) {
        ragTaskRunner->unregisterCallback();
        ragTaskRunner->stop = true;
        ragTaskRunner = nullptr;
        lifuren::RAGService::getInstance().deleteRAGTask(path);
    }
    if(documentMarkConfig.size() > 0) {
        pathPtr->value(pathPtr->find_index(documentMarkConfig.begin()->path.c_str()));
        pathPtr->redraw();
        reloadConfig(windowPtr, pathPtr->text());
    } else {
        pathPtr->value(-1);
        windowPtr->redrawConfigElement();
    }
}

static void markCallback(Fl_Widget*, void* voidPtr) {
    auto& ragService = lifuren::RAGService::getInstance();
    lifuren::RAGTask task{
        pathPtr->text(),
        ragPtr->text(),
        chunkPtr->text(),
        embeddingPtr->text(),
    };
    ragTaskRunner = ragService.buildRAGTask(task);
    ragTaskRunner->registerCallback(percentCallback);
}

static void stopCallback(Fl_Widget*, void* voidPtr) {
    if(ragTaskRunner) {
        ragTaskRunner->unregisterCallback();
        ragTaskRunner->stop = true;
        ragTaskRunner = nullptr;
    }
}

static bool reloadConfig(lifuren::DocumentMarkWindow* windowPtr, const std::string& path) {
    bool newPath = false;
    auto& documentMarkConfig = lifuren::config::CONFIG.documentMark;
    auto iterator = std::find(documentMarkConfig.begin(), documentMarkConfig.end(), path);
    if(iterator == documentMarkConfig.end()) {
        lifuren::config::DocumentMarkConfig config{ path, "", "", "" };
        ::documentMarkConfig = &documentMarkConfig.emplace_back(config);
        newPath = true;
    } else {
        ::documentMarkConfig = &*iterator;
        newPath = false;
    }
    windowPtr->redrawConfigElement();
    if(ragTaskRunner) {
        ragTaskRunner->unregisterCallback();
        ragTaskRunner->stop = true;
        ragTaskRunner = nullptr;
    }
    auto& ragService = lifuren::RAGService::getInstance();
    ragTaskRunner = ragService.getRAGTask(path);
    if(ragTaskRunner) {
        ragTaskRunner->registerCallback(percentCallback);
    }
    return newPath;
}

static void percentCallback(float percent, bool done) {
    SPDLOG_DEBUG("当前任务进度：{} - {}", percent, done);
}

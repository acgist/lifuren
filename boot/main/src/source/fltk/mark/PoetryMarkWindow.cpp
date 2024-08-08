#include "lifuren/FLTK.hpp"

#include "spdlog/spdlog.h"

#include "FL/Fl_Input.H"
#include "FL/Fl_Button.H"
#include "FL/Fl_Text_Buffer.H"
#include "FL/Fl_Text_Display.H"

#include "lifuren/Jsons.hpp"
#include "lifuren/FLTKWidget.hpp"
#include "lifuren/model/Poetry.hpp"

// TODO: 问题双击选中中文崩溃

// 旧的路径
static std::string oldPath;
// 诗词列表
static nlohmann::json poetryJson;
// 当前诗词索引
static nlohmann::json::iterator poetryIterator;
// 诗词文件列表
static std::vector<std::string> fileVector;
// 当前诗词文件索引
static std::vector<std::string>::iterator fileIterator;

static Fl_Button* prevPtr{ nullptr };
static Fl_Button* nextPtr{ nullptr };
static Fl_Input*  datasetPathPtr{ nullptr };
static Fl_Button* autoMarkPtr   { nullptr };
static Fl_Text_Buffer*  sourceBufferPtr   { nullptr };
static Fl_Text_Display* sourceDisplayPtr  { nullptr };
static Fl_Text_Buffer*  rhythmicBufferPtr { nullptr };
static Fl_Text_Display* rhythmicDisplayPtr{ nullptr };
static Fl_Text_Buffer*  targetBufferPtr   { nullptr };
static Fl_Text_Display* targetDisplayPtr  { nullptr };

/**
 * 加载诗词
 * 
 * @param path 文件路径
 */
static void loadFileVector(const std::string& path);
// 加载诗词列表
static void loadPoetryJson();
// 匹配诗词规则
static void matchPoetryRhythmic();
// 上首诗词
static void prevPoetry(Fl_Widget*, void*);
// 下首诗词
static void nextPoetry(Fl_Widget*, void*);

lifuren::PoetryMarkWindow::PoetryMarkWindow(int width, int height, const char* title) : MarkWindow(width, height, title) {
    this->poetryMarkConfigPtr = &lifuren::config::CONFIG.poetryMark;
}

lifuren::PoetryMarkWindow::~PoetryMarkWindow() {
    SPDLOG_DEBUG("关闭窗口：{}", __FILE__);
    lifuren::config::saveFile();
    // 静态资源
    LFR_DELETE_PTR(prevPtr);
    LFR_DELETE_PTR(nextPtr);
    LFR_DELETE_PTR(datasetPathPtr);
    LFR_DELETE_PTR(autoMarkPtr);
    LFR_DELETE_PTR(sourceDisplayPtr);
    LFR_DELETE_PTR(sourceBufferPtr);
    LFR_DELETE_PTR(rhythmicDisplayPtr);
    LFR_DELETE_PTR(rhythmicBufferPtr);
    LFR_DELETE_PTR(targetDisplayPtr);
    LFR_DELETE_PTR(targetBufferPtr);
    // 清理数据
    oldPath = "";
    fileVector.clear();
    poetryJson.clear();
}

void lifuren::PoetryMarkWindow::drawElement() {
    // 配置按钮
    datasetPathPtr = new lifuren::Fl_Input_Directory_Chooser(100, 10, this->w() - 200, 30, "数据目录");
    datasetPathPtr->value(this->poetryMarkConfigPtr->datasetPath.c_str());
    prevPtr = new Fl_Button(10,  50, 100, 30, "上首诗词");
    nextPtr = new Fl_Button(120, 50, 100, 30, "下首诗词");
    autoMarkPtr   = new Fl_Button(230, 50, 100, 30, "自动匹配");
    LFR_INPUT_DIRECTORY_CHOOSER_CALLBACK(datasetPathPtr, poetryMarkConfigPtr, datasetPath, PoetryMarkWindow, loadFileVector);
    // 诗词
    sourceDisplayPtr = new Fl_Text_Display(10, 110, (this->w() - 40) / 3, this->h() - 120, "诗词");
    sourceDisplayPtr->wrap_mode(sourceDisplayPtr->WRAP_AT_COLUMN, sourceDisplayPtr->textfont());
    sourceDisplayPtr->color(FL_BACKGROUND_COLOR);
    sourceBufferPtr = new Fl_Text_Buffer();
    sourceDisplayPtr->buffer(sourceBufferPtr);
    // 格律
    rhythmicDisplayPtr = new Fl_Text_Display(20 + (this->w() - 40) / 3, 110, (this->w() - 40) / 3, this->h() - 120, "格律");
    rhythmicDisplayPtr->wrap_mode(rhythmicDisplayPtr->WRAP_AT_COLUMN, rhythmicDisplayPtr->textfont());
    rhythmicDisplayPtr->color(FL_BACKGROUND_COLOR);
    rhythmicBufferPtr = new Fl_Text_Buffer();
    rhythmicDisplayPtr->buffer(rhythmicBufferPtr);
    // 分词
    targetDisplayPtr = new Fl_Text_Display(30 + (this->w() - 40) / 3 * 2, 110, (this->w() - 40) / 3, this->h() - 120, "分词");
    targetDisplayPtr->wrap_mode(targetDisplayPtr->WRAP_AT_COLUMN, targetDisplayPtr->textfont());
    targetDisplayPtr->color(FL_BACKGROUND_COLOR);
    targetBufferPtr = new Fl_Text_Buffer();
    targetDisplayPtr->buffer(targetBufferPtr);
    // 事件
    prevPtr->callback(prevPoetry, this);
    nextPtr->callback(nextPoetry, this);
    // 加载资源
    loadFileVector(this->poetryMarkConfigPtr->datasetPath);
}

static void prevPoetry(Fl_Widget* widgetPtr, void* voidPtr) {
    if(fileVector.empty()) {
        SPDLOG_DEBUG("没有诗词文件：{}", oldPath);
        return;
    }
    if (poetryIterator == poetryJson.begin()) {
        if(fileIterator == fileVector.begin()) {
            fileIterator = fileVector.end();
        }
        --fileIterator;
        loadPoetryJson();
        poetryIterator = poetryJson.end();
    }
    --poetryIterator;
    matchPoetryRhythmic();
}

static void nextPoetry(Fl_Widget* widgetPtr, void* voidPtr) {
    if(fileVector.empty()) {
        SPDLOG_DEBUG("没有诗词文件：{}", oldPath);
        return;
    }
    ++poetryIterator;
    if(poetryIterator == poetryJson.end()) {
        ++fileIterator;
        if(fileIterator == fileVector.end()) {
            fileIterator = fileVector.begin();
        }
        loadPoetryJson();
    }
    matchPoetryRhythmic();
}

static void loadFileVector(const std::string& path) {
    if(path.empty() || path == oldPath) {
        SPDLOG_DEBUG("忽略诗词目录加载：{}", path);
        return;
    }
    oldPath = path;
    fileVector.clear();
    lifuren::files::listFiles(fileVector, oldPath, { ".json" });
    fileIterator = fileVector.begin();
    loadPoetryJson();
    matchPoetryRhythmic();
}

static void loadPoetryJson() {
    if(fileIterator == fileVector.end()) {
        SPDLOG_WARN("没有可用诗词目录");
        return;
    }
    std::string json = lifuren::files::loadFile(*fileIterator);
    poetryJson = nlohmann::json::parse(json);
    poetryIterator = poetryJson.begin();
}

static void matchPoetryRhythmic() {
    if(fileIterator == fileVector.end()) {
        SPDLOG_WARN("没有可用诗词目录");
        return;
    }
    if(poetryIterator == poetryJson.end()) {
        SPDLOG_WARN("没有可用诗词");
        return;
    }
    lifuren::Poetry poetry = *poetryIterator;
    poetry.preproccess();
    SPDLOG_DEBUG("解析诗词：{} - {}", *fileIterator, poetry.title);
    // 原始内容
    sourceBufferPtr->text(poetry.title.c_str());
    sourceBufferPtr->append("\n");
    sourceBufferPtr->append(poetry.author.c_str());
    sourceBufferPtr->append("\n");
    sourceBufferPtr->append("\n");
    sourceBufferPtr->append(poetry.segment.c_str());
    sourceBufferPtr->append("\n");
    sourceBufferPtr->append("\n");
    sourceBufferPtr->append(poetry.simpleSegment.c_str());
    sourceDisplayPtr->redraw();
    // 匹配规则
    const bool hasLabel = poetry.matchLabel();
    // 规则内容
    if(hasLabel) {
        const lifuren::LabelText* labelPtr = poetry.label;
        rhythmicBufferPtr->text(labelPtr->name.c_str());
        rhythmicBufferPtr->append("\n");
        rhythmicBufferPtr->append("\n");
        rhythmicBufferPtr->append(lifuren::poetry::beautify(labelPtr->example).c_str());
    } else {
        rhythmicBufferPtr->text("没有匹配规则");
    }
    rhythmicDisplayPtr->redraw();
    // 分词内容
    if(hasLabel) {
        poetry.participle();
        targetBufferPtr->text(poetry.title.c_str());
        targetBufferPtr->append("\n");
        targetBufferPtr->append("\n");
        targetBufferPtr->append(poetry.participleSegment.c_str());
    } else {
        targetBufferPtr->text("没有匹配规则");
    }
    targetDisplayPtr->redraw();
}

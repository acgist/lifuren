#include "lifuren/FLTK.hpp"

#include "spdlog/spdlog.h"

#include "FL/Fl_Input.H"
#include "FL/Fl_Button.H"
#include "FL/Fl_Choice.H"
#include "FL/Fl_Text_Buffer.H"
#include "FL/Fl_Text_Display.H"

#include "lifuren/Jsons.hpp"
#include "lifuren/Strings.hpp"
#include "lifuren/FLTKWidget.hpp"
#include "lifuren/model/Poetry.hpp"

// TODO: 问题双击选中中文崩溃

// 诗词列表
static nlohmann::json poetryJson;
// 当前诗词索引
static nlohmann::json::iterator poetryIterator;
// 诗词文件列表
static std::vector<std::string> fileVector;
// 当前诗词文件索引
static std::vector<std::string>::iterator fileIterator;
static lifuren::config::PoetryMarkConfig* poetryMarkConfig{ nullptr };

static Fl_Button* newPtr     { nullptr };
static Fl_Choice* pathPtr    { nullptr };
static Fl_Button* deletePtr  { nullptr };
static Fl_Button* prevPtr    { nullptr };
static Fl_Button* nextPtr    { nullptr };
static Fl_Button* autoMarkPtr{ nullptr };
static Fl_Text_Buffer*  sourceBufferPtr   { nullptr };
static Fl_Text_Display* sourceDisplayPtr  { nullptr };
static Fl_Text_Buffer*  rhythmicBufferPtr { nullptr };
static Fl_Text_Display* rhythmicDisplayPtr{ nullptr };
static Fl_Text_Buffer*  targetBufferPtr   { nullptr };
static Fl_Text_Display* targetDisplayPtr  { nullptr };

static void newCallback   (Fl_Widget*, void*);
static void pathCallback  (Fl_Widget*, void*);
static void deleteCallback(Fl_Widget*, void*);
static bool reloadConfig(lifuren::PoetryMarkWindow*, const std::string&);
static void prevPoetry(Fl_Widget*, void*);
static void nextPoetry(Fl_Widget*, void*);
static void loadFileVector(const std::string& path);
static void loadPoetryJson();
static void matchPoetryRhythmic();

lifuren::PoetryMarkWindow::PoetryMarkWindow(int width, int height, const char* title) : MarkWindow(width, height, title) {
}

lifuren::PoetryMarkWindow::~PoetryMarkWindow() {
    SPDLOG_DEBUG("关闭窗口：{}", __FILE__);
    // 保存配置
    this->saveConfig();
    // 清理数据
    poetryMarkConfig = nullptr;
    fileVector.clear();
    poetryJson.clear();
    // 静态资源
    LFR_DELETE_PTR(pathPtr);
    LFR_DELETE_PTR(newPtr);
    LFR_DELETE_PTR(deletePtr);
    LFR_DELETE_PTR(prevPtr);
    LFR_DELETE_PTR(nextPtr);
    LFR_DELETE_PTR(autoMarkPtr);
    LFR_DELETE_PTR(sourceDisplayPtr);
    LFR_DELETE_PTR(sourceBufferPtr);
    LFR_DELETE_PTR(rhythmicDisplayPtr);
    LFR_DELETE_PTR(rhythmicBufferPtr);
    LFR_DELETE_PTR(targetDisplayPtr);
    LFR_DELETE_PTR(targetBufferPtr);
}

void lifuren::PoetryMarkWindow::saveConfig() {
    lifuren::Configuration::saveConfig();
}

void lifuren::PoetryMarkWindow::redrawConfigElement() {
}

void lifuren::PoetryMarkWindow::drawElement() {
    // 配置按钮
    pathPtr     = new Fl_Choice(80,  10, 200, 30, "诗词目录");
    newPtr      = new Fl_Button(280, 10, 100, 30, "新增目录");
    deletePtr   = new Fl_Button(380, 10, 100, 30, "删除目录");
    prevPtr     = new Fl_Button(80,  50, 100, 30, "上首诗词");
    nextPtr     = new Fl_Button(190, 50, 100, 30, "下首诗词");
    autoMarkPtr = new Fl_Button(300, 50, 100, 30, "自动匹配");
    // 诗词
    sourceDisplayPtr = new Fl_Text_Display(10, 110, (this->w() - 40) / 3, this->h() - 120, "诗词");
    sourceDisplayPtr->wrap_mode(sourceDisplayPtr->WRAP_AT_COLUMN, sourceDisplayPtr->textfont());
    sourceDisplayPtr->color(FL_BACKGROUND_COLOR);
    sourceBufferPtr = new Fl_Text_Buffer();
    sourceDisplayPtr->buffer(sourceBufferPtr);
    sourceDisplayPtr->end();
    // 格律
    rhythmicDisplayPtr = new Fl_Text_Display(20 + (this->w() - 40) / 3, 110, (this->w() - 40) / 3, this->h() - 120, "格律");
    rhythmicDisplayPtr->wrap_mode(rhythmicDisplayPtr->WRAP_AT_COLUMN, rhythmicDisplayPtr->textfont());
    rhythmicDisplayPtr->color(FL_BACKGROUND_COLOR);
    rhythmicBufferPtr = new Fl_Text_Buffer();
    rhythmicDisplayPtr->buffer(rhythmicBufferPtr);
    rhythmicDisplayPtr->end();
    // 分词
    targetDisplayPtr = new Fl_Text_Display(30 + (this->w() - 40) / 3 * 2, 110, (this->w() - 40) / 3, this->h() - 120, "分词");
    targetDisplayPtr->wrap_mode(targetDisplayPtr->WRAP_AT_COLUMN, targetDisplayPtr->textfont());
    targetDisplayPtr->color(FL_BACKGROUND_COLOR);
    targetBufferPtr = new Fl_Text_Buffer();
    targetDisplayPtr->buffer(targetBufferPtr);
    targetDisplayPtr->end();
    // 事件
    // 诗词目录
    const auto& poetryMark = lifuren::config::CONFIG.poetryMark;
    for(auto& value : poetryMark) {
        std::string path = value.path;
        #if _WIN32
        lifuren::strings::replace(path, "\\", "\\\\");
        #else
        lifuren::strings::replace(path, "/", "\\/");
        #endif
        pathPtr->add(path.c_str());
    }
    pathPtr->callback(pathCallback, this);
    // 新增目录
    newPtr->callback(newCallback, this);
    // 删除目录
    deletePtr->callback(deleteCallback, this);
    // 上首诗词
    prevPtr->callback(prevPoetry, this);
    // 下首诗词
    nextPtr->callback(nextPoetry, this);
    // 加载资源
    // loadFileVector(this->poetryMarkConfigPtr->datasetPath);
}

static void newCallback(Fl_Widget*, void* voidPtr) {
    std::string filename = lifuren::directoryChooser("选择诗词目录");
    if(filename.empty()) {
        return;
    }
    lifuren::PoetryMarkWindow* windowPtr = static_cast<lifuren::PoetryMarkWindow*>(voidPtr);
    if(reloadConfig(windowPtr, filename)) {
        std::string path = filename;
        #if _WIN32
        lifuren::strings::replace(path, "\\", "\\\\");
        #else
        lifuren::strings::replace(path, "/", "\\/");
        #endif
        pathPtr->add(path.c_str());
    }
    int index = pathPtr->find_index(filename.c_str());
    pathPtr->value(index);
}

static void pathCallback(Fl_Widget*, void* voidPtr) {
    lifuren::PoetryMarkWindow* windowPtr = static_cast<lifuren::PoetryMarkWindow*>(voidPtr);
    windowPtr->saveConfig();
    reloadConfig(windowPtr, pathPtr->text());
}

static void deleteCallback(Fl_Widget*, void* voidPtr) {
    int index = pathPtr->value();
    if(index < 0) {
        return;
    }
    lifuren::PoetryMarkWindow* windowPtr = static_cast<lifuren::PoetryMarkWindow*>(voidPtr);
    auto& poetryMarkConfig = lifuren::config::CONFIG.poetryMark;
    auto iterator = std::find(poetryMarkConfig.begin(), poetryMarkConfig.end(), pathPtr->text());
    if(iterator != poetryMarkConfig.end()) {
        poetryMarkConfig.erase(iterator);
    }
    pathPtr->remove(index);
    ::poetryMarkConfig = nullptr;
    if(poetryMarkConfig.size() > 0) {
        index = pathPtr->find_index(poetryMarkConfig.begin()->path.c_str());
        pathPtr->value(index);
        pathPtr->redraw();
        reloadConfig(windowPtr, pathPtr->text());
    } else {
        pathPtr->value(-1);
        windowPtr->redrawConfigElement();
    }
}

static bool reloadConfig(lifuren::PoetryMarkWindow* windowPtr, const std::string& path) {
    bool newPath = false;
    auto& poetryMarkConfig = lifuren::config::CONFIG.poetryMark;
    auto iterator = std::find(poetryMarkConfig.begin(), poetryMarkConfig.end(), path);
    if(iterator == poetryMarkConfig.end()) {
        lifuren::config::PoetryMarkConfig config{};
        config.path = path;
        ::poetryMarkConfig = &poetryMarkConfig.emplace_back(config);
        newPath = true;
    } else {
        ::poetryMarkConfig = &*iterator;
        newPath = false;
    }
    windowPtr->redrawConfigElement();
    return newPath;
}

static void prevPoetry(Fl_Widget* widgetPtr, void* voidPtr) {
    if(!poetryMarkConfig) {
        return;
    }
    if(fileVector.empty()) {
        SPDLOG_DEBUG("没有诗词文件：{}", poetryMarkConfig->path);
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
    if(!poetryMarkConfig) {
        return;
    }
    if(fileVector.empty()) {
        SPDLOG_DEBUG("没有诗词文件：{}", poetryMarkConfig->path);
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
    if(!poetryMarkConfig) {
        return;
    }
    if(path.empty()) {
        SPDLOG_DEBUG("忽略诗词目录加载：{}", path);
        return;
    }
    fileVector.clear();
    lifuren::files::listFiles(fileVector, poetryMarkConfig->path, { ".json" });
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

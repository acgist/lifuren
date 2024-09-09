/**
 * 诗词JSON文件格式
 * [
 *     {
 *         "title"     : "标题",
 *         "author"    : "作者",
 *         "rhythmic"  : "格律|词牌",
 *         "paragraphs": [ "段落", ... ]
 *     },
 *     ...
 * ]
 * 
 * TODO:
 * 1. 中文双击崩溃问题
 */
#include "lifuren/FLTK.hpp"

#include "spdlog/spdlog.h"

#include "Fl/fl_ask.H"
#include "FL/Fl_Button.H"
#include "FL/Fl_Choice.H"
#include "FL/Fl_Text_Buffer.H"
#include "FL/Fl_Text_Display.H"

#include "lifuren/Jsons.hpp"
#include "lifuren/Strings.hpp"
#include "lifuren/RAGClient.hpp"
#include "lifuren/model/Poetry.hpp"

static nlohmann::json           poetryJson    {};
static nlohmann::json::iterator poetryIterator{};
static std::vector<std::string>           fileVector  {};
static std::vector<std::string>::iterator fileIterator{};
static lifuren::config::PoetryMarkConfig* poetryMarkConfig{ nullptr };

static Fl_Button* newPtr     { nullptr };
static Fl_Choice* pathPtr    { nullptr };
static Fl_Button* deletePtr  { nullptr };
static Fl_Button* prevPtr    { nullptr };
static Fl_Button* nextPtr    { nullptr };
static Fl_Button* autoMarkPtr{ nullptr };
static Fl_Button* ragPtr     { nullptr };
static Fl_Button* modelPtr   { nullptr };
static Fl_Button* sdModelPtr { nullptr };

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
static void autoMark  (Fl_Widget*, void*);
static void loadFileVector(const std::string& path);
static void loadPoetryJson();
static void matchPoetryRhythmic();
static void resetPoetryRhythmic();
static void ragCallback  (Fl_Widget*, void*);
static void modelCallback(Fl_Widget*, void*);

lifuren::PoetryMarkWindow::PoetryMarkWindow(int width, int height, const char* title) : MarkWindow(width, height, title) {
}

lifuren::PoetryMarkWindow::~PoetryMarkWindow() {
    // 保存配置
    this->saveConfig();
    // 清理数据
    poetryMarkConfig = nullptr;
    fileVector.clear();
    poetryJson.clear();
    // 静态资源
    LFR_DELETE_PTR(newPtr);
    LFR_DELETE_PTR(pathPtr);
    LFR_DELETE_PTR(deletePtr);
    LFR_DELETE_PTR(prevPtr);
    LFR_DELETE_PTR(nextPtr);
    LFR_DELETE_PTR(autoMarkPtr);
    LFR_DELETE_PTR(modelPtr);
    LFR_DELETE_PTR(sdModelPtr);
    LFR_DELETE_PTR(ragPtr);
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
    // 绘制界面
    pathPtr     = new Fl_Choice(10,  10, 300, 30);
    newPtr      = new Fl_Button(310, 10, 100, 30, "新增目录");
    deletePtr   = new Fl_Button(410, 10, 100, 30, "删除目录");
    prevPtr     = new Fl_Button(10,  50, 100, 30, "上首诗词");
    nextPtr     = new Fl_Button(110, 50, 100, 30, "下首诗词");
    autoMarkPtr = new Fl_Button(210, 50, 100, 30, "匹配格律");
    autoMarkPtr = new Fl_Button(310, 50, 100, 30, "匹配图片");
    ragPtr      = new Fl_Button(410, 50, 100, 30, "建立索引");
    modelPtr    = new Fl_Button(510, 50, 100, 30, "训练诗词模型");
    sdModelPtr  = new Fl_Button(610, 50, 100, 30, "微调图片模型");
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
    // 绑定事件
    // 诗词目录
    const auto& poetryMark = lifuren::config::CONFIG.poetryMark;
    for(auto& value : poetryMark) {
        std::string path = value.path;
        LFR_CHOICE_TRANSFER(path);
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
    // 自动匹配
    autoMarkPtr->callback(autoMark, this);
    // 建立索引
    ragPtr->callback(ragCallback, this);
    // 训练模型
    modelPtr->callback(modelCallback, this);
}

static void newCallback(Fl_Widget*, void* voidPtr) {
    std::string filename = lifuren::directoryChooser("选择诗词目录");
    if(filename.empty()) {
        return;
    }
    lifuren::PoetryMarkWindow* windowPtr = static_cast<lifuren::PoetryMarkWindow*>(voidPtr);
    if(reloadConfig(windowPtr, filename)) {
        std::string path = filename;
        LFR_CHOICE_TRANSFER(path);
        pathPtr->add(path.c_str());
    }
    pathPtr->value(pathPtr->find_index(filename.c_str()));
}

static void pathCallback(Fl_Widget*, void* voidPtr) {
    lifuren::PoetryMarkWindow* windowPtr = static_cast<lifuren::PoetryMarkWindow*>(voidPtr);
    windowPtr->saveConfig();
    reloadConfig(windowPtr, pathPtr->text());
}

static void deleteCallback(Fl_Widget*, void* voidPtr) {
    const int index = pathPtr->value();
    if(index < 0) {
        return;
    }
    lifuren::PoetryMarkWindow* windowPtr = static_cast<lifuren::PoetryMarkWindow*>(voidPtr);
    auto& poetryMarkConfig = lifuren::config::CONFIG.poetryMark;
    auto iterator = std::find(poetryMarkConfig.begin(), poetryMarkConfig.end(), pathPtr->text());
    if(iterator != poetryMarkConfig.end()) {
        poetryMarkConfig.erase(iterator);
    }
    auto& ragService = lifuren::RAGService::getInstance();
    ragService.deleteRAGTask(pathPtr->text());
    pathPtr->remove(index);
    ::poetryMarkConfig = nullptr;
    resetPoetryRhythmic();
    if(poetryMarkConfig.size() > 0) {
        pathPtr->value(pathPtr->find_index(poetryMarkConfig.begin()->path.c_str()));
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
        lifuren::config::PoetryMarkConfig config{ path };
        ::poetryMarkConfig = &poetryMarkConfig.emplace_back(config);
        newPath = true;
    } else {
        ::poetryMarkConfig = &*iterator;
        newPath = false;
    }
    windowPtr->redrawConfigElement();
    loadFileVector(path);
    return newPath;
}

static void prevPoetry(Fl_Widget* widgetPtr, void* voidPtr) {
    if(!poetryMarkConfig) {
        return;
    }
    if(fileVector.empty()) {
        resetPoetryRhythmic();
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
        resetPoetryRhythmic();
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

static void autoMark(Fl_Widget*, void*) {
    if(!poetryMarkConfig) {
        fl_message("没有选择目录");
        return;
    }
    auto stopFile = fileIterator;
    lifuren::Poetry stopPoetry = *poetryIterator;
    while(true) {
        ++poetryIterator;
        if(poetryIterator == poetryJson.end()) {
            ++fileIterator;
            if(fileIterator == fileVector.end()) {
                fileIterator = fileVector.begin();
            }
            loadPoetryJson();
        }
        lifuren::Poetry poetry = *poetryIterator;
        if(stopFile == fileIterator && stopPoetry == poetry) {
            fl_message("所有诗词全部匹配");
            break;
        }
        poetry.preproccess();
        const bool hasLabel = poetry.matchLabel();
        if(hasLabel) {
        } else {
            matchPoetryRhythmic();
            break;
        }
    }
}

static void loadFileVector(const std::string& path) {
    if(!poetryMarkConfig) {
        return;
    }
    if(path.empty()) {
        resetPoetryRhythmic();
        SPDLOG_DEBUG("忽略诗词目录加载：{}", path);
        return;
    }
    SPDLOG_DEBUG("加载诗词目录：{}", path);
    fileVector.clear();
    lifuren::files::listFiles(fileVector, poetryMarkConfig->path, { ".json" });
    fileIterator = fileVector.begin();
    loadPoetryJson();
    matchPoetryRhythmic();
}

static void loadPoetryJson() {
    if(fileIterator == fileVector.end()) {
        resetPoetryRhythmic();
        SPDLOG_WARN("没有可用诗词目录");
        return;
    }
    SPDLOG_DEBUG("加载诗词文件：{}", *fileIterator);
    std::string&& json = lifuren::files::loadFile(*fileIterator);
    poetryJson     = nlohmann::json::parse(json);
    poetryIterator = poetryJson.begin();
}

static void matchPoetryRhythmic() {
    if(fileIterator == fileVector.end()) {
        resetPoetryRhythmic();
        SPDLOG_WARN("没有可用诗词目录");
        return;
    }
    if(poetryIterator == poetryJson.end()) {
        resetPoetryRhythmic();
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

static void resetPoetryRhythmic() {
    sourceBufferPtr->text("");
    rhythmicBufferPtr->text("");
    targetBufferPtr->text("");
}

static void ragCallback(Fl_Widget*, void*) {
    const int index = pathPtr->value();
    if(index < 0) {
        return;
    }
    auto& ragService = lifuren::RAGService::getInstance();
    if(ragService.getRAGTask(pathPtr->text())) {
        ragService.stopRAGTask(pathPtr->text());
    } else {
        ragService.buildRAGTask(pathPtr->text());
    }
}

static void modelCallback(Fl_Widget*, void*) {
}

    // SPDLOG_DEBUG("预览图片：{}", *imageIterator);
    // LFR_DELETE_PTR(previewImagePtr);
    // Fl_Shared_Image* previewSharedPtr = Fl_Shared_Image::get((*imageIterator).c_str());
    // if(previewSharedPtr->num_images() <= 0) {
    //     fl_message("图片读取失败");
    //     resetImage();
    //     SPDLOG_WARN("图片加载失败：{}", *imageIterator);
    //     // previewSharedPtr->release();
    //     return;
    // }
    // const int boxWidth    = previewBoxPtr->w();
    // const int boxHeight   = previewBoxPtr->h();
    // const int imageWidth  = previewSharedPtr->w();
    // const int imageHeight = previewSharedPtr->h();
    // double scale;
    // if(imageWidth * boxHeight > imageHeight * boxWidth) {
    //     scale = LFR_IMAGE_PREVIEW_SCALE * imageWidth / boxWidth;
    // } else {
    //     scale = LFR_IMAGE_PREVIEW_SCALE * imageHeight / boxHeight;
    // }
    // previewImagePtr = previewSharedPtr->copy((int) (imageWidth / scale), (int) (imageHeight / scale));
    // previewSharedPtr->release();
    // previewBoxPtr->image(previewImagePtr);
    // previewBoxPtr->redraw();

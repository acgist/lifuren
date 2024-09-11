/**
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

#include "lifuren/Ptr.hpp"
#include "lifuren/RAG.hpp"
#include "lifuren/Jsons.hpp"
#include "lifuren/Config.hpp"
#include "lifuren/Poetrys.hpp"
#include "lifuren/Strings.hpp"

static nlohmann::json           poetryJson    {};
static nlohmann::json::iterator poetryIterator{};
static std::vector<std::string>           fileVector  {};
static std::vector<std::string>::iterator fileIterator{};

static lifuren::config::MarkConfig* selectMarkConfig{ nullptr };

static Fl_Button* newPtr     { nullptr };
static Fl_Choice* pathPtr    { nullptr };
static Fl_Button* deletePtr  { nullptr };
static Fl_Button* prevPtr    { nullptr };
static Fl_Button* nextPtr    { nullptr };
static Fl_Button* autoMarkPtr{ nullptr };
static Fl_Button* imagePtr   { nullptr };
static Fl_Button* ragTaskPtr { nullptr };
static Fl_Button* modelPtr   { nullptr };
static Fl_Button* sdModelPtr { nullptr };

static Fl_Text_Buffer*  sourceBufferPtr { nullptr };
static Fl_Text_Display* sourceDisplayPtr{ nullptr };
static Fl_Text_Buffer*  rhythmBufferPtr { nullptr };
static Fl_Text_Display* rhythmDisplayPtr{ nullptr };
static Fl_Text_Buffer*  targetBufferPtr { nullptr };
static Fl_Text_Display* targetDisplayPtr{ nullptr };

static void newCallback   (Fl_Widget*, void*);
static void pathCallback  (Fl_Widget*, void*);
static void deleteCallback(Fl_Widget*, void*);
static bool reloadConfig(lifuren::MarkWindow*, const std::string&);
static void prevPoetry(Fl_Widget*, void*);
static void nextPoetry(Fl_Widget*, void*);
static void autoMark  (Fl_Widget*, void*);
static void loadFileVector(const std::string& path);
static void loadPoetryJson();
static void matchPoetryRhythm();
static void resetPoetryRhythm();
static void ragTaskCallback(Fl_Widget*, void*);
static void modelCallback(Fl_Widget*, void*);

static const char* PATH_UNCHOISE_MESSAGE = "没有选择数据目录";

lifuren::MarkWindow::MarkWindow(int width, int height, const char* title) : Window(width, height, title) {
}

lifuren::MarkWindow::~MarkWindow() {
    // 保存配置
    this->saveConfig();
    // 清理数据
    selectMarkConfig = nullptr;
    fileVector.clear();
    poetryJson.clear();
    // 静态资源
    LFR_DELETE_PTR(newPtr);
    LFR_DELETE_PTR(pathPtr);
    LFR_DELETE_PTR(deletePtr);
    LFR_DELETE_PTR(prevPtr);
    LFR_DELETE_PTR(nextPtr);
    LFR_DELETE_PTR(autoMarkPtr);
    LFR_DELETE_PTR(imagePtr);
    LFR_DELETE_PTR(modelPtr);
    LFR_DELETE_PTR(sdModelPtr);
    LFR_DELETE_PTR(ragTaskPtr);
    LFR_DELETE_PTR(sourceDisplayPtr);
    LFR_DELETE_PTR(sourceBufferPtr);
    LFR_DELETE_PTR(rhythmDisplayPtr);
    LFR_DELETE_PTR(rhythmBufferPtr);
    LFR_DELETE_PTR(targetDisplayPtr);
    LFR_DELETE_PTR(targetBufferPtr);
}

void lifuren::MarkWindow::saveConfig() {
    lifuren::Configuration::saveConfig();
}

void lifuren::MarkWindow::redrawConfigElement() {
}

void lifuren::MarkWindow::drawElement() {
    // 绘制界面
    pathPtr     = new Fl_Choice(10,  10, 300, 30);
    newPtr      = new Fl_Button(310, 10, 100, 30, "新增目录");
    deletePtr   = new Fl_Button(410, 10, 100, 30, "删除目录");
    prevPtr     = new Fl_Button(10,  50, 100, 30, "上首诗词");
    nextPtr     = new Fl_Button(110, 50, 100, 30, "下首诗词");
    autoMarkPtr = new Fl_Button(210, 50, 100, 30, "匹配格律");
    imagePtr    = new Fl_Button(310, 50, 100, 30, "匹配图片");
    ragTaskPtr  = new Fl_Button(410, 50, 100, 30, "建立索引");
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
    rhythmDisplayPtr = new Fl_Text_Display(20 + (this->w() - 40) / 3, 110, (this->w() - 40) / 3, this->h() - 120, "格律");
    rhythmDisplayPtr->wrap_mode(rhythmDisplayPtr->WRAP_AT_COLUMN, rhythmDisplayPtr->textfont());
    rhythmDisplayPtr->color(FL_BACKGROUND_COLOR);
    rhythmBufferPtr = new Fl_Text_Buffer();
    rhythmDisplayPtr->buffer(rhythmBufferPtr);
    rhythmDisplayPtr->end();
    // 分词
    targetDisplayPtr = new Fl_Text_Display(30 + (this->w() - 40) / 3 * 2, 110, (this->w() - 40) / 3, this->h() - 120, "分词");
    targetDisplayPtr->wrap_mode(targetDisplayPtr->WRAP_AT_COLUMN, targetDisplayPtr->textfont());
    targetDisplayPtr->color(FL_BACKGROUND_COLOR);
    targetBufferPtr = new Fl_Text_Buffer();
    targetDisplayPtr->buffer(targetBufferPtr);
    targetDisplayPtr->end();
    // 绑定事件
    // 诗词目录
    const auto& mark = lifuren::config::CONFIG.mark;
    for(auto& value : mark) {
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
    ragTaskPtr->callback(ragTaskCallback, this);
    // 训练模型
    modelPtr->callback(modelCallback, this);
}

static void newCallback(Fl_Widget*, void* voidPtr) {
    std::string filename = lifuren::directoryChooser("选择诗词目录");
    if(filename.empty()) {
        return;
    }
    lifuren::MarkWindow* windowPtr = static_cast<lifuren::MarkWindow*>(voidPtr);
    if(reloadConfig(windowPtr, filename)) {
        std::string path = filename;
        LFR_CHOICE_TRANSFER(path);
        pathPtr->add(path.c_str());
    }
    pathPtr->value(pathPtr->find_index(filename.c_str()));
}

static void pathCallback(Fl_Widget*, void* voidPtr) {
    lifuren::MarkWindow* windowPtr = static_cast<lifuren::MarkWindow*>(voidPtr);
    windowPtr->saveConfig();
    reloadConfig(windowPtr, pathPtr->text());
}

static void deleteCallback(Fl_Widget*, void* voidPtr) {
    const int index = pathPtr->value();
    if(index < 0) {
        return;
    }
    lifuren::MarkWindow* windowPtr = static_cast<lifuren::MarkWindow*>(voidPtr);
    auto& markConfig = lifuren::config::CONFIG.mark;
    auto iterator = std::find(markConfig.begin(), markConfig.end(), pathPtr->text());
    if(iterator != markConfig.end()) {
        markConfig.erase(iterator);
    }
    auto& ragService = lifuren::RAGService::getInstance();
    ragService.deleteRAGTask(pathPtr->text());
    pathPtr->remove(index);
    selectMarkConfig = nullptr;
    resetPoetryRhythm();
    if(markConfig.size() > 0) {
        pathPtr->value(pathPtr->find_index(markConfig.begin()->path.c_str()));
        pathPtr->redraw();
        reloadConfig(windowPtr, pathPtr->text());
    } else {
        pathPtr->value(-1);
        windowPtr->redrawConfigElement();
    }
}

static bool reloadConfig(lifuren::MarkWindow* windowPtr, const std::string& path) {
    bool newPath = false;
    auto& markConfig = lifuren::config::CONFIG.mark;
    auto iterator = std::find(markConfig.begin(), markConfig.end(), path);
    if(iterator == markConfig.end()) {
        lifuren::config::MarkConfig config{ path };
        selectMarkConfig = &markConfig.emplace_back(config);
        newPath = true;
    } else {
        selectMarkConfig = &*iterator;
        newPath = false;
    }
    windowPtr->redrawConfigElement();
    loadFileVector(path);
    return newPath;
}

static void prevPoetry(Fl_Widget* widgetPtr, void* voidPtr) {
    if(!selectMarkConfig) {
        return;
    }
    if(fileVector.empty()) {
        resetPoetryRhythm();
        SPDLOG_DEBUG("没有诗词文件：{}", selectMarkConfig->path);
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
    matchPoetryRhythm();
}

static void nextPoetry(Fl_Widget* widgetPtr, void* voidPtr) {
    if(!selectMarkConfig) {
        return;
    }
    if(fileVector.empty()) {
        resetPoetryRhythm();
        SPDLOG_DEBUG("没有诗词文件：{}", selectMarkConfig->path);
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
    matchPoetryRhythm();
}

static void autoMark(Fl_Widget*, void*) {
    if(!selectMarkConfig) {
        fl_message(PATH_UNCHOISE_MESSAGE);
        return;
    }
    auto stopFile = fileIterator;
    lifuren::poetrys::Poetry stopPoetry = *poetryIterator;
    while(true) {
        ++poetryIterator;
        if(poetryIterator == poetryJson.end()) {
            ++fileIterator;
            if(fileIterator == fileVector.end()) {
                fileIterator = fileVector.begin();
            }
            loadPoetryJson();
        }
        lifuren::poetrys::Poetry poetry = *poetryIterator;
        if(stopFile == fileIterator && stopPoetry == poetry) {
            fl_message("所有诗词全部匹配");
            break;
        }
        poetry.preproccess();
        const bool hasRhythm = poetry.matchRhythm();
        if(hasRhythm) {
        } else {
            matchPoetryRhythm();
            break;
        }
    }
}

static void loadFileVector(const std::string& path) {
    if(!selectMarkConfig) {
        return;
    }
    if(path.empty()) {
        resetPoetryRhythm();
        SPDLOG_DEBUG("忽略诗词目录加载：{}", path);
        return;
    }
    SPDLOG_DEBUG("加载诗词目录：{}", path);
    fileVector.clear();
    lifuren::files::listFiles(fileVector, selectMarkConfig->path, { ".json" });
    fileIterator = fileVector.begin();
    loadPoetryJson();
    matchPoetryRhythm();
}

static void loadPoetryJson() {
    if(fileIterator == fileVector.end()) {
        resetPoetryRhythm();
        SPDLOG_WARN("没有可用诗词目录");
        return;
    }
    SPDLOG_DEBUG("加载诗词文件：{}", *fileIterator);
    std::string&& json = lifuren::files::loadFile(*fileIterator);
    poetryJson     = nlohmann::json::parse(json);
    poetryIterator = poetryJson.begin();
}

static void matchPoetryRhythm() {
    if(fileIterator == fileVector.end()) {
        resetPoetryRhythm();
        SPDLOG_WARN("没有可用诗词目录");
        return;
    }
    if(poetryIterator == poetryJson.end()) {
        resetPoetryRhythm();
        SPDLOG_WARN("没有可用诗词");
        return;
    }
    lifuren::poetrys::Poetry poetry = *poetryIterator;
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
    const bool hasRhythm = poetry.matchRhythm();
    // 规则内容
    if(hasRhythm) {
        const lifuren::config::Rhythm* rhythmPtr = poetry.rhythmPtr;
        rhythmBufferPtr->text(rhythmPtr->name.c_str());
        rhythmBufferPtr->append("\n");
        rhythmBufferPtr->append("\n");
        rhythmBufferPtr->append(lifuren::poetrys::beautify(rhythmPtr->example).c_str());
    } else {
        rhythmBufferPtr->text("没有匹配规则");
    }
    rhythmDisplayPtr->redraw();
    // 分词内容
    if(hasRhythm) {
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

static void resetPoetryRhythm() {
    sourceBufferPtr->text("");
    rhythmBufferPtr->text("");
    targetBufferPtr->text("");
}

static void ragTaskCallback(Fl_Widget*, void*) {
    if(pathPtr->value() < 0) {
        fl_message(PATH_UNCHOISE_MESSAGE);
        return;
    }
    const std::string path = pathPtr->text();
    auto& ragService = lifuren::RAGService::getInstance();
    if(ragService.getRAGTask(path) && fl_choice("任务已经开始是否取消？", "否", "是", 0) == 1) {
        ragService.stopRAGTask(pathPtr->text());
        return;
    }
    ragService.buildRAGTask(pathPtr->text());
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

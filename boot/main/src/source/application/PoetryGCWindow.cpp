#include "../../header/Window.hpp"

#include "Jsons.hpp"
#include "Poetry.hpp"

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

// 原始诗词内容
static Fl_Text_Buffer* sourceBufferPtr = nullptr;
// 原始诗词
static Fl_Text_Display* sourceDisplayPtr = nullptr;
// 格律内容
static Fl_Text_Buffer* rhythmicBufferPtr = nullptr;
// 格律
static Fl_Text_Display* rhythmicDisplayPtr = nullptr;
// 目标诗词内容
static Fl_Text_Buffer* targetBufferPtr = nullptr;
// 目标诗词
static Fl_Text_Display* targetDisplayPtr = nullptr;

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

lifuren::PoetryGCWindow::PoetryGCWindow(int width, int height, const char* title) : ModelGCWindow(width, height, title) {
    auto iterator = SETTINGS.find("PoetryGC");
    if(iterator == SETTINGS.end()) {
        this->settingPtr = new Setting();
        SETTINGS.insert(std::make_pair("PoetryGC", *this->settingPtr));
    } else {
        this->settingPtr = &iterator->second;
    }
}

lifuren::PoetryGCWindow::~PoetryGCWindow() {
    SPDLOG_DEBUG("关闭窗口：{}", __FILE__);
    lifuren::jsons::saveFile(SETTINGS_PATH, lifuren::SETTINGS);
    LFR_DELETE_THIS_PTR(modelPathPtr);
    LFR_DELETE_THIS_PTR(datasetPathPtr);
    LFR_DELETE_THIS_PTR(prevPtr);
    LFR_DELETE_THIS_PTR(nextPtr);
    LFR_DELETE_THIS_PTR(trainStartPtr);
    LFR_DELETE_THIS_PTR(trainStopPtr);
    LFR_DELETE_THIS_PTR(generatePtr);
    LFR_DELETE_THIS_PTR(autoMarkPtr);
    // 静态资源
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

void lifuren::PoetryGCWindow::drawElement() {
    this->modelPathPtr = new Fl_Input_Directory_Chooser(100, 10, this->w() - 200, 30, "模型目录");
    this->modelPathPtr->value(this->settingPtr->modelPath.c_str());
    this->datasetPathPtr = new Fl_Input_Directory_Chooser(100, 50, this->w() - 200, 30, "数据目录");
    this->datasetPathPtr->value(this->settingPtr->datasetPath.c_str());
    LFR_INPUT_DIRECTORY_CHOOSER_CALLBACK(modelPathPtr, modelPath, PoetryGCWindow);
    LFR_INPUT_DIRECTORY_CHOOSER_CALLBACK_CALL(datasetPathPtr, datasetPath, PoetryGCWindow, loadFileVector);
    this->prevPtr       = new Fl_Button(10,  90, 100, 30, "上首诗词");
    this->nextPtr       = new Fl_Button(120, 90, 100, 30, "下首诗词");
    this->autoMarkPtr   = new Fl_Button(230, 90, 100, 30, "自动匹配");
    this->trainStartPtr = new Fl_Button(340, 90, 100, 30, "开始训练");
    this->trainStopPtr  = new Fl_Button(450, 90, 100, 30, "结束训练");
    this->generatePtr   = new Fl_Button(560, 90, 100, 30, "生成诗词");
    // 诗词
    sourceDisplayPtr = new Fl_Text_Display(10, 150, (this->w() - 40) / 3, this->h() - 160, "诗词");
    sourceDisplayPtr->wrap_mode(sourceDisplayPtr->WRAP_AT_COLUMN, sourceDisplayPtr->textfont());
    sourceDisplayPtr->color(FL_BACKGROUND_COLOR);
    sourceBufferPtr = new Fl_Text_Buffer();
    sourceDisplayPtr->buffer(sourceBufferPtr);
    // 格律
    rhythmicDisplayPtr = new Fl_Text_Display(20 + (this->w() - 40) / 3, 150, (this->w() - 40) / 3, this->h() - 160, "格律");
    rhythmicDisplayPtr->wrap_mode(rhythmicDisplayPtr->WRAP_AT_COLUMN, rhythmicDisplayPtr->textfont());
    rhythmicDisplayPtr->color(FL_BACKGROUND_COLOR);
    rhythmicBufferPtr = new Fl_Text_Buffer();
    rhythmicDisplayPtr->buffer(rhythmicBufferPtr);
    // 分词
    targetDisplayPtr = new Fl_Text_Display(30 + (this->w() - 40) / 3 * 2, 150, (this->w() - 40) / 3, this->h() - 160, "分词");
    targetDisplayPtr->wrap_mode(targetDisplayPtr->WRAP_AT_COLUMN, targetDisplayPtr->textfont());
    targetDisplayPtr->color(FL_BACKGROUND_COLOR);
    targetBufferPtr = new Fl_Text_Buffer();
    targetDisplayPtr->buffer(targetBufferPtr);
    // 事件
    this->prevPtr->callback(prevPoetry, this);
    this->nextPtr->callback(nextPoetry, this);
    // 加载资源
    loadFileVector(this->settingPtr->datasetPath);
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
        SPDLOG_DEBUG("忽略诗词目录加载：{} - {}", __func__, path);
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
    std::string json = lifuren::files::loadFile(*fileIterator);
    poetryJson = nlohmann::json::parse(json);
    poetryIterator = poetryJson.begin();
}

static void matchPoetryRhythmic() {
    if(poetryIterator == poetryJson.end()) {
        SPDLOG_WARN("没有可用诗词");
        return;
    }
    nlohmann::json::iterator titleIterator    = poetryIterator->find("title");
    nlohmann::json::iterator rhythmicIterator = poetryIterator->find("rhythmic");
    lifuren::Poetry poetry;
    if(titleIterator == poetryIterator->end()) {
        // 诗
        poetry = *poetryIterator;
    } else if(rhythmicIterator == poetryIterator->end()) {
        // 词
        poetry = *poetryIterator;
    } else {
        SPDLOG_DEBUG("匹配不到诗词规则：{}", poetryIterator->dump());
        return;
    }
    poetry.preproccess();
    SPDLOG_DEBUG("解析诗词：{} - {} - {}", __func__, *fileIterator, poetry.title);
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

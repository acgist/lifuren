#include "../../header/Window.hpp"

#include "Jsons.hpp"

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

/**
 * 加载诗词
 * 
 * @param path 文件路径
 */
static void loadFileVector(const std::string& path);
// 加载诗词列表
static void loadPoetryJson();

// 上首诗词
static void prevPoetry(Fl_Widget*, void*);
// 下首诗词
static void nextPoetry(Fl_Widget*, void*);
// 匹配规则
static void matchRule(Fl_Widget*, void*);

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
    LFR_DELETE_THIS_PTR(ruleDisplayPtr);
    LFR_DELETE_THIS_PTR(ruleBufferPtr);
    LFR_DELETE_THIS_PTR(sourceDisplayPtr);
    LFR_DELETE_THIS_PTR(sourceBufferPtr);
    LFR_DELETE_THIS_PTR(targetDisplayPtr);
    LFR_DELETE_THIS_PTR(targetBufferPtr);
}

void lifuren::PoetryGCWindow::drawElement() {
    this->modelPathPtr = new Fl_Input_Directory_Chooser(100, 10, this->w() - 200, 30, "模型目录");
    this->modelPathPtr->value(this->settingPtr->modelPath.c_str());
    this->datasetPathPtr = new Fl_Input_Directory_Chooser(100, 50, this->w() - 200, 30, "数据目录");
    this->datasetPathPtr->value(this->settingPtr->datasetPath.c_str());
    LFR_INPUT_DIRECTORY_CHOOSER_CALLBACK(modelPathPtr, modelPath, PoetryGCWindow);
    LFR_INPUT_DIRECTORY_CHOOSER_CALLBACK(datasetPathPtr, datasetPath, PoetryGCWindow);
    this->prevPtr       = new Fl_Button(10,  90, 100, 30, "上首诗词");
    this->nextPtr       = new Fl_Button(120, 90, 100, 30, "下首诗词");
    this->autoMarkPtr   = new Fl_Button(230, 90, 100, 30, "自动匹配");
    this->trainStartPtr = new Fl_Button(340, 90, 100, 30, "开始训练");
    this->trainStopPtr  = new Fl_Button(450, 90, 100, 30, "结束训练");
    this->generatePtr   = new Fl_Button(560, 90, 100, 30, "生成诗词");
    // 原始
    this->sourceDisplayPtr = new Fl_Text_Display(10, 150, (this->w() - 40) / 3, this->h() - 160, "原始");
    this->sourceDisplayPtr->wrap_mode(this->sourceDisplayPtr->WRAP_AT_COLUMN, this->sourceDisplayPtr->textfont());
    this->sourceDisplayPtr->color(FL_BACKGROUND_COLOR);
    this->sourceBufferPtr = new Fl_Text_Buffer();
    this->sourceDisplayPtr->buffer(this->sourceBufferPtr);
    // 规则
    this->ruleDisplayPtr = new Fl_Text_Display(20 + (this->w() - 40) / 3, 150, (this->w() - 40) / 3, this->h() - 160, "规则");
    this->ruleDisplayPtr->wrap_mode(this->ruleDisplayPtr->WRAP_AT_COLUMN, this->ruleDisplayPtr->textfont());
    this->ruleDisplayPtr->color(FL_BACKGROUND_COLOR);
    this->ruleBufferPtr = new Fl_Text_Buffer();
    this->ruleDisplayPtr->buffer(this->ruleBufferPtr);
    // 目标
    this->targetDisplayPtr = new Fl_Text_Display(30 + (this->w() - 40) / 3 * 2, 150, (this->w() - 40) / 3, this->h() - 160, "目标");
    this->targetDisplayPtr->wrap_mode(this->targetDisplayPtr->WRAP_AT_COLUMN, this->targetDisplayPtr->textfont());
    this->targetDisplayPtr->color(FL_BACKGROUND_COLOR);
    this->targetBufferPtr = new Fl_Text_Buffer();
    this->targetDisplayPtr->buffer(this->targetBufferPtr);
    // 事件
    this->prevPtr->callback(prevPoetry, this);
    this->nextPtr->callback(nextPoetry, this);
}

static void prevPoetry(Fl_Widget* widgetPtr, void* voidPtr) {
    lifuren::PoetryGCWindow* windowPtr = (lifuren::PoetryGCWindow*) voidPtr;
    loadFileVector(windowPtr->datasetPath());
    if(fileVector.empty()) {
        SPDLOG_DEBUG("没有诗词文件：{}", windowPtr->datasetPath());
        return;
    }
    if(poetryIterator == poetryJson.begin()) {
        if(fileIterator == fileVector.begin()) {
            fileIterator = fileVector.end();
        }
        --fileIterator;
        loadPoetryJson();
    }
    --poetryIterator;
    windowPtr->sourceBufferPtr->text("1");
}

static void nextPoetry(Fl_Widget* widgetPtr, void* voidPtr) {
    lifuren::PoetryGCWindow* windowPtr = (lifuren::PoetryGCWindow*) voidPtr;
    loadFileVector(windowPtr->datasetPath());
    if(fileVector.empty()) {
        SPDLOG_DEBUG("没有诗词文件：{}", windowPtr->datasetPath());
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
    windowPtr->sourceBufferPtr->text("2");
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
}

static void loadPoetryJson() {
    std::string json = lifuren::files::loadFile(*fileIterator);
    poetryJson = nlohmann::json::parse(json);
    poetryIterator = poetryJson.begin();
}

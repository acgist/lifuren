/**
 * TODO:
 * 1. 添加图片缩放
 * 2. 添加图片反转
 */
#include "lifuren/FLTK.hpp"

#include <list>

#include "spdlog/spdlog.h"

#include "lifuren/Files.hpp"
#include "lifuren/Strings.hpp"
#include "lifuren/FLTKWidget.hpp"
#include "lifuren/config/Label.hpp"

#include "FL/fl_ask.H"
#include "FL/Fl_Box.H"
#include "FL/Fl_Input.H"
#include "FL/Fl_Button.H"
#include "FL/Fl_Choice.H"
#include "FL/Fl_Text_Buffer.H"
#include "FL/Fl_Text_Editor.H"
#include "Fl/Fl_Shared_Image.H"

#include "nlohmann/json.hpp"

// 图片缩放
#ifndef LFR_IMAGE_PREVIEW_SCALE
#define LFR_IMAGE_PREVIEW_SCALE 1.2
#endif

#ifndef LFR_CHOICE_ADD_LIST_PROXY
#define LFR_CHOICE_ADD_LIST_PROXY(x, y, map, choicePtr, groupName, labelName, defaultValue) \
LFR_CHOICE_ADD_LIST(x, y, map, choicePtr, groupName, labelName, defaultValue)               \
choiceList.push_back(choicePtr);
#endif

// 头部
static Fl_Choice* fasePtr    { nullptr };
static Fl_Choice* faxingPtr  { nullptr };
static Fl_Choice* meimaoPtr  { nullptr };
static Fl_Choice* yanjingPtr { nullptr };
static Fl_Choice* biziPtr    { nullptr };
static Fl_Choice* yachiPtr   { nullptr };
static Fl_Choice* zuibaPtr   { nullptr };
static Fl_Choice* kouhongPtr { nullptr };
static Fl_Choice* biaoqingPtr{ nullptr };
static Fl_Choice* lianxingPtr{ nullptr };
// 上身
static Fl_Choice* rufangPtr  { nullptr };
static Fl_Choice* shouxingPtr{ nullptr };
static Fl_Choice* yaobuPtr   { nullptr };
// 下身
static Fl_Choice* tunbuPtr  { nullptr };
static Fl_Choice* tuixingPtr{ nullptr };
// 衣着
static Fl_Choice* sediaoPtr{ nullptr };
static Fl_Choice* yifuPtr  { nullptr };
static Fl_Choice* kuziPtr  { nullptr };
static Fl_Choice* xieziPtr { nullptr };
// 饰品
static Fl_Choice* toushiPtr { nullptr };
static Fl_Choice* ershiPtr  { nullptr };
static Fl_Choice* yanshiPtr { nullptr };
static Fl_Choice* lianshiPtr{ nullptr };
static Fl_Choice* shoushiPtr{ nullptr };
static Fl_Choice* baobaoPtr { nullptr };
// 特征
static Fl_Choice* fusePtr    { nullptr };
static Fl_Choice* pifuPtr    { nullptr };
static Fl_Choice* xinggePtr  { nullptr };
static Fl_Choice* nianlingPtr{ nullptr };
static Fl_Choice* shengaoPtr { nullptr };
static Fl_Choice* tixingPtr  { nullptr };
static Fl_Choice* titaiPtr   { nullptr };
static Fl_Choice* zhiyePtr   { nullptr };
// 环境
static Fl_Choice* tianqiPtr  { nullptr };
static Fl_Choice* qianjingPtr{ nullptr };
static Fl_Choice* beijingPtr { nullptr };

// 功能按钮
static Fl_Button* newPtr   { nullptr };
static Fl_Choice* pathPtr  { nullptr };
static Fl_Button* deletePtr{ nullptr };
static Fl_Button* prevPtr  { nullptr };
static Fl_Button* nextPtr  { nullptr };
static Fl_Button* markPtr  { nullptr };
static Fl_Button* resetPtr { nullptr };
static Fl_Box*    previewBoxPtr  { nullptr };
static Fl_Image*  previewImagePtr{ nullptr };
static Fl_Text_Buffer* moreBufferPtr{ nullptr };
static Fl_Text_Editor* moreEditorPtr{ nullptr };

static std::vector<std::string>           imageVector{};
static std::vector<std::string>::iterator imageIterator{};
static std::list<Fl_Choice*> choiceList{};
static lifuren::config::ImageMarkConfig* imageMarkConfig{ nullptr };

static void newCallback   (Fl_Widget*, void*);
static void pathCallback  (Fl_Widget*, void*);
static void deleteCallback(Fl_Widget*, void*);
static bool reloadConfig(lifuren::ImageMarkWindow*, const std::string&);
static void prevImage  (Fl_Widget*, void*);
static void nextImage  (Fl_Widget*, void*);
static void markImage  (Fl_Widget*, void*);
static void resetConfig(Fl_Widget*, void*);
static void resetImage();
static void remarkConfig();
static void previewImage();
static void loadImageVector(const std::string& path);

lifuren::ImageMarkWindow::ImageMarkWindow(int width, int height, const char* title) : MarkWindow(width, height, title) {
}

lifuren::ImageMarkWindow::~ImageMarkWindow() {
    SPDLOG_DEBUG("关闭窗口：{}", __FILE__);
    // 保存配置
    this->saveConfig();
    // 清理数据
    imageMarkConfig = nullptr;
    for(auto ptr : choiceList) {
        delete ptr;
    }
    choiceList.clear();
    imageVector.clear();
    // 释放资源
    LFR_DELETE_PTR(newPtr);
    LFR_DELETE_PTR(pathPtr);
    LFR_DELETE_PTR(deletePtr);
    LFR_DELETE_PTR(prevPtr);
    LFR_DELETE_PTR(nextPtr);
    LFR_DELETE_PTR(markPtr);
    LFR_DELETE_PTR(resetPtr);
    LFR_DELETE_PTR(previewBoxPtr);
    LFR_DELETE_PTR(previewImagePtr);
    LFR_DELETE_PTR(moreEditorPtr);
    LFR_DELETE_PTR(moreBufferPtr);
}

void lifuren::ImageMarkWindow::saveConfig() {
    lifuren::Configuration::saveConfig();
}

void lifuren::ImageMarkWindow::redrawConfigElement() {
}

void lifuren::ImageMarkWindow::drawElement() {
    // 配置按钮
    pathPtr   = new Fl_Choice(80,  10, 200, 30, "图片目录");
    newPtr    = new Fl_Button(280, 10, 100, 30, "新增目录");
    deletePtr = new Fl_Button(380, 10, 100, 30, "删除目录");
    prevPtr   = new Fl_Button(80,  50, 100, 30, "上张图片");
    nextPtr   = new Fl_Button(190, 50, 100, 30, "下张图片");
    markPtr   = new Fl_Button(300, 50, 100, 30, "标记图片");
    resetPtr  = new Fl_Button(410, 50, 100, 30, "重置选项");
    // 图片预览
    previewBoxPtr = new Fl_Box(this->w() / 2 + 200, this->h() / 2 - 150, 400, 300, "预览图片");
    previewBoxPtr->box(FL_FLAT_BOX);
    // 头部设置
    int yPos = 90;
    LFR_CHOICE_ADD_LIST_PROXY(40,  yPos, LABEL_IMAGE, fasePtr,     "头部", "发色", "默认");
    LFR_CHOICE_ADD_LIST_PROXY(160, yPos, LABEL_IMAGE, faxingPtr,   "头部", "发型", "默认");
    LFR_CHOICE_ADD_LIST_PROXY(280, yPos, LABEL_IMAGE, meimaoPtr,   "头部", "眉毛", "默认");
    LFR_CHOICE_ADD_LIST_PROXY(400, yPos, LABEL_IMAGE, yanjingPtr,  "头部", "眼睛", "默认");
    LFR_CHOICE_ADD_LIST_PROXY(520, yPos, LABEL_IMAGE, biziPtr,     "头部", "鼻子", "默认");
    LFR_CHOICE_ADD_LIST_PROXY(640, yPos, LABEL_IMAGE, yachiPtr,    "头部", "牙齿", "默认");
    yPos += 40;
    LFR_CHOICE_ADD_LIST_PROXY(40,  yPos, LABEL_IMAGE, zuibaPtr,    "头部", "嘴巴", "默认");
    LFR_CHOICE_ADD_LIST_PROXY(160, yPos, LABEL_IMAGE, kouhongPtr,  "头部", "口红", "默认");
    LFR_CHOICE_ADD_LIST_PROXY(280, yPos, LABEL_IMAGE, biaoqingPtr, "头部", "表情", "默认");
    LFR_CHOICE_ADD_LIST_PROXY(400, yPos, LABEL_IMAGE, lianxingPtr, "头部", "脸型", "默认");
    // 上身设置
    yPos += 40;
    LFR_CHOICE_ADD_LIST_PROXY(40,  yPos, LABEL_IMAGE, rufangPtr  , "上身", "乳房", "默认");
    LFR_CHOICE_ADD_LIST_PROXY(160, yPos, LABEL_IMAGE, shouxingPtr, "上身", "手型", "默认");
    LFR_CHOICE_ADD_LIST_PROXY(280, yPos, LABEL_IMAGE, yaobuPtr   , "上身", "腰部", "默认");
    // 下身设置
    yPos += 40;
    LFR_CHOICE_ADD_LIST_PROXY(40,  yPos, LABEL_IMAGE, tunbuPtr  , "下身", "臀部", "默认");
    LFR_CHOICE_ADD_LIST_PROXY(160, yPos, LABEL_IMAGE, tuixingPtr, "下身", "腿型", "默认");
    // 衣着设置
    yPos += 40;
    LFR_CHOICE_ADD_LIST_PROXY(40,  yPos, LABEL_IMAGE, sediaoPtr, "衣着", "色调", "默认");
    LFR_CHOICE_ADD_LIST_PROXY(160, yPos, LABEL_IMAGE, yifuPtr  , "衣着", "衣服", "默认");
    LFR_CHOICE_ADD_LIST_PROXY(280, yPos, LABEL_IMAGE, kuziPtr  , "衣着", "裤子", "默认");
    LFR_CHOICE_ADD_LIST_PROXY(400, yPos, LABEL_IMAGE, xieziPtr , "衣着", "鞋子", "默认");
    // 饰品设置
    yPos += 40;
    LFR_CHOICE_ADD_LIST_PROXY(40,  yPos, LABEL_IMAGE, toushiPtr , "饰品", "头饰", "默认");
    LFR_CHOICE_ADD_LIST_PROXY(160, yPos, LABEL_IMAGE, ershiPtr  , "饰品", "耳饰", "默认");
    LFR_CHOICE_ADD_LIST_PROXY(280, yPos, LABEL_IMAGE, yanshiPtr , "饰品", "眼饰", "默认");
    LFR_CHOICE_ADD_LIST_PROXY(400, yPos, LABEL_IMAGE, lianshiPtr, "饰品", "脸饰", "默认");
    LFR_CHOICE_ADD_LIST_PROXY(520, yPos, LABEL_IMAGE, shoushiPtr, "饰品", "首饰", "默认");
    LFR_CHOICE_ADD_LIST_PROXY(640, yPos, LABEL_IMAGE, baobaoPtr , "饰品", "包包", "默认");
    // 特征设置
    yPos += 40;
    LFR_CHOICE_ADD_LIST_PROXY(40,  yPos, LABEL_IMAGE, fusePtr    , "特征", "肤色", "默认");
    LFR_CHOICE_ADD_LIST_PROXY(160, yPos, LABEL_IMAGE, pifuPtr    , "特征", "皮肤", "默认");
    LFR_CHOICE_ADD_LIST_PROXY(280, yPos, LABEL_IMAGE, xinggePtr  , "特征", "性格", "默认");
    LFR_CHOICE_ADD_LIST_PROXY(400, yPos, LABEL_IMAGE, nianlingPtr, "特征", "年龄", "默认");
    LFR_CHOICE_ADD_LIST_PROXY(520, yPos, LABEL_IMAGE, shengaoPtr , "特征", "身高", "默认");
    LFR_CHOICE_ADD_LIST_PROXY(640, yPos, LABEL_IMAGE, tixingPtr  , "特征", "体型", "默认");
    yPos += 40;
    LFR_CHOICE_ADD_LIST_PROXY(40,  yPos, LABEL_IMAGE, titaiPtr   , "特征", "姿态", "默认");
    LFR_CHOICE_ADD_LIST_PROXY(160, yPos, LABEL_IMAGE, zhiyePtr   , "特征", "职业", "默认");
    // 环境设置
    yPos += 40;
    LFR_CHOICE_ADD_LIST_PROXY(40,  yPos, LABEL_IMAGE, tianqiPtr  , "环境", "天气", "默认");
    LFR_CHOICE_ADD_LIST_PROXY(160, yPos, LABEL_IMAGE, qianjingPtr, "环境", "前景", "默认");
    LFR_CHOICE_ADD_LIST_PROXY(280, yPos, LABEL_IMAGE, beijingPtr , "环境", "背景", "默认");
    // 更多设置
    yPos += 40;
    moreEditorPtr = new Fl_Text_Editor(10, yPos, 720, 100);
    moreBufferPtr = new Fl_Text_Buffer();
    moreEditorPtr->buffer(moreBufferPtr);
    moreEditorPtr->wrap_mode(moreEditorPtr->WRAP_AT_COLUMN, moreEditorPtr->textfont());
    moreEditorPtr->end();
    // 事件
    // 图片目录
    const auto& imageMark = lifuren::config::CONFIG.imageMark;
    for(auto& value : imageMark) {
        std::string path = value.path;
        LFR_CHOICE_TRANSFER(path);
        pathPtr->add(path.c_str());
    }
    pathPtr->callback(pathCallback, this);
    // 新增目录
    newPtr->callback(newCallback, this);
    // 删除目录
    deletePtr->callback(deleteCallback, this);
    // 上张图片
    prevPtr->callback(prevImage, this);
    // 下张图片
    nextPtr->callback(nextImage, this);
    // 标记图片
    markPtr->callback(markImage, this);
    // 重置选项
    resetPtr->callback(resetConfig, this);
}

static void newCallback(Fl_Widget*, void* voidPtr) {
    std::string filename = lifuren::directoryChooser("选择图片目录");
    if(filename.empty()) {
        return;
    }
    lifuren::ImageMarkWindow* windowPtr = static_cast<lifuren::ImageMarkWindow*>(voidPtr);
    if(reloadConfig(windowPtr, filename)) {
        std::string path = filename;
        LFR_CHOICE_TRANSFER(path);
        pathPtr->add(path.c_str());
    }
    pathPtr->value(pathPtr->find_index(filename.c_str()));
}

static void pathCallback(Fl_Widget*, void* voidPtr) {
    lifuren::ImageMarkWindow* windowPtr = static_cast<lifuren::ImageMarkWindow*>(voidPtr);
    windowPtr->saveConfig();
    reloadConfig(windowPtr, pathPtr->text());
}

static void deleteCallback(Fl_Widget*, void* voidPtr) {
    const int index = pathPtr->value();
    if(index < 0) {
        return;
    }
    lifuren::ImageMarkWindow* windowPtr = static_cast<lifuren::ImageMarkWindow*>(voidPtr);
    auto& imageMarkConfig = lifuren::config::CONFIG.imageMark;
    auto iterator = std::find(imageMarkConfig.begin(), imageMarkConfig.end(), pathPtr->text());
    if(iterator != imageMarkConfig.end()) {
        imageMarkConfig.erase(iterator);
    }
    resetImage();
    pathPtr->remove(index);
    ::imageMarkConfig = nullptr;
    if(imageMarkConfig.size() > 0) {
        pathPtr->value(pathPtr->find_index(imageMarkConfig.begin()->path.c_str()));
        pathPtr->redraw();
        reloadConfig(windowPtr, pathPtr->text());
    } else {
        pathPtr->value(-1);
        windowPtr->redrawConfigElement();
    }
}

static bool reloadConfig(lifuren::ImageMarkWindow* windowPtr, const std::string& path) {
    bool newPath = false;
    auto& imageMarkConfig = lifuren::config::CONFIG.imageMark;
    auto iterator = std::find(imageMarkConfig.begin(), imageMarkConfig.end(), path);
    if(iterator == imageMarkConfig.end()) {
        lifuren::config::ImageMarkConfig config{ path };
        ::imageMarkConfig = &imageMarkConfig.emplace_back(config);
        newPath = true;
    } else {
        ::imageMarkConfig = &*iterator;
        newPath = false;
    }
    windowPtr->redrawConfigElement();
    loadImageVector(path);
    return newPath;
}

static void prevImage(Fl_Widget* widgetPtr, void* voidPtr) {
    if(!imageMarkConfig) {
        return;
    }
    if(imageVector.empty()) {
        SPDLOG_DEBUG("没有图片文件：{}", imageMarkConfig->path);
        return;
    }
    if(imageIterator == imageVector.begin()) {
        imageIterator = imageVector.end();
    }
    --imageIterator;
    previewImage();
}

static void nextImage(Fl_Widget* widgetPtr, void* voidPtr) {
    if(!imageMarkConfig) {
        return;
    }
    if(imageVector.empty()) {
        SPDLOG_DEBUG("没有图片文件：{}", imageMarkConfig->path);
        return;
    }
    ++imageIterator;
    if(imageIterator == imageVector.end()) {
        imageIterator = imageVector.begin();
    }
    previewImage();
}

static void markImage(Fl_Widget*, void*) {
    if(!imageMarkConfig) {
        return;
    }
    nlohmann::json mark{};
    for(auto ptr : choiceList) {
        if(ptr->value() <= 0) {
            continue;
        }
        mark[ptr->label()] = ptr->text();
    }
    const char* more = moreBufferPtr->text();
    if(std::strlen(more) > 0) {
        mark["描述"] = more;
    }
    std::filesystem::path path      = imageMarkConfig->path;
    std::filesystem::path imagePath = *imageIterator;
    path = path / "index" / (imagePath.filename().string() + ".mark");
    lifuren::files::saveFile(path.string(), mark.dump());
}

static void resetConfig(Fl_Widget*, void*) {
    for(auto ptr : choiceList) {
        ptr->value(0);
    }
    moreBufferPtr->text("");
}

static void resetImage() {
    LFR_DELETE_PTR(previewImagePtr);
    previewBoxPtr->image(nullptr);
    previewBoxPtr->redraw();
}

static void remarkConfig() {
    if(!imageMarkConfig) {
        return;
    }
    std::filesystem::path path      = imageMarkConfig->path;
    std::filesystem::path imagePath = *imageIterator;
    path = path / "index" / (imagePath.filename().string() + ".mark");
    const auto&& json = lifuren::files::loadFile(path.string());
    if(json.empty()) {
        resetConfig(nullptr, nullptr);
        return;
    }
    nlohmann::json mark = nlohmann::json::parse(json);
    for(auto ptr : choiceList) {
        auto iterator = mark.find(ptr->label());
        if(iterator != mark.end()) {
            ptr->value(ptr->find_index(iterator->get<std::string>().c_str()));
        }
    }
    auto iterator = mark.find("描述");
    if(iterator != mark.end()) {
        moreBufferPtr->text(iterator->get<std::string>().c_str());
    }
}

static void previewImage() {
    if(imageIterator == imageVector.end()) {
        resetImage();
        SPDLOG_WARN("没有可用图片");
        return;
    }
    SPDLOG_DEBUG("预览图片：{}", *imageIterator);
    LFR_DELETE_PTR(previewImagePtr);
    Fl_Shared_Image* previewSharedPtr = Fl_Shared_Image::get((*imageIterator).c_str());
    if(previewSharedPtr->num_images() <= 0) {
        fl_message("图片读取失败");
        resetImage();
        SPDLOG_WARN("图片加载失败：{}", *imageIterator);
        // previewSharedPtr->release();
        return;
    }
    const int boxWidth    = previewBoxPtr->w();
    const int boxHeight   = previewBoxPtr->h();
    const int imageWidth  = previewSharedPtr->w();
    const int imageHeight = previewSharedPtr->h();
    double scale;
    if(imageWidth * boxHeight > imageHeight * boxWidth) {
        scale = LFR_IMAGE_PREVIEW_SCALE * imageWidth / boxWidth;
    } else {
        scale = LFR_IMAGE_PREVIEW_SCALE * imageHeight / boxHeight;
    }
    previewImagePtr = previewSharedPtr->copy((int) (imageWidth / scale), (int) (imageHeight / scale));
    previewSharedPtr->release();
    previewBoxPtr->image(previewImagePtr);
    previewBoxPtr->redraw();
    remarkConfig();
}

static void loadImageVector(const std::string& path) {
    if(path.empty()) {
        SPDLOG_DEBUG("忽略图片目录加载：{}", path);
        return;
    }
    resetImage();
    imageVector.clear();
    lifuren::files::listFiles(imageVector, path, { ".jpg", ".jpeg", ".png" });
    imageIterator = imageVector.begin();
    previewImage();
}

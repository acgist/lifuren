#include "lifuren/FLTK.hpp"

#include <list>

#include "spdlog/spdlog.h"

#include "lifuren/Files.hpp"
#include "lifuren/FLTKWidget.hpp"
#include "lifuren/config/Label.hpp"

#include "FL/Fl_Box.H"
#include "FL/Fl_Input.H"
#include "FL/Fl_Button.H"
#include "FL/Fl_Choice.H"
#include "FL/Fl_Text_Buffer.H"
#include "FL/Fl_Text_Editor.H"
#include "Fl/Fl_Shared_Image.H"

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
static Fl_Choice* fasePtr     = nullptr;
static Fl_Choice* faxingPtr   = nullptr;
static Fl_Choice* meimaoPtr   = nullptr;
static Fl_Choice* yanjingPtr  = nullptr;
static Fl_Choice* biziPtr     = nullptr;
static Fl_Choice* yachiPtr    = nullptr;
static Fl_Choice* zuibaPtr    = nullptr;
static Fl_Choice* kouhongPtr  = nullptr;
static Fl_Choice* biaoqingPtr = nullptr;
static Fl_Choice* lianxingPtr = nullptr;
// 上身
static Fl_Choice* rufangPtr   = nullptr;
static Fl_Choice* shouxingPtr = nullptr;
static Fl_Choice* yaobuPtr    = nullptr;
// 下身
static Fl_Choice* tunbuPtr   = nullptr;
static Fl_Choice* tuixingPtr = nullptr;
// 衣着
static Fl_Choice* sediaoPtr = nullptr;
static Fl_Choice* yifuPtr   = nullptr;
static Fl_Choice* kuziPtr   = nullptr;
static Fl_Choice* xieziPtr  = nullptr;
// 饰品
static Fl_Choice* toushiPtr  = nullptr;
static Fl_Choice* ershiPtr   = nullptr;
static Fl_Choice* yanshiPtr  = nullptr;
static Fl_Choice* lianshiPtr = nullptr;
static Fl_Choice* shoushiPtr = nullptr;
static Fl_Choice* baobaoPtr  = nullptr;
// 特征
static Fl_Choice* fusePtr     = nullptr;
static Fl_Choice* pifuPtr     = nullptr;
static Fl_Choice* xinggePtr   = nullptr;
static Fl_Choice* nianlingPtr = nullptr;
static Fl_Choice* shengaoPtr  = nullptr;
static Fl_Choice* tixingPtr   = nullptr;
static Fl_Choice* titaiPtr    = nullptr;
static Fl_Choice* zhiyePtr    = nullptr;
// 环境
static Fl_Choice* tianqiPtr   = nullptr;
static Fl_Choice* qianjingPtr = nullptr;
static Fl_Choice* beijingPtr  = nullptr;

// 功能按钮
static Fl_Button* prevPtr { nullptr };
static Fl_Button* nextPtr { nullptr };
static Fl_Button* resetPtr{ nullptr };
static Fl_Input*  datasetPathPtr { nullptr };
static Fl_Box*    previewBoxPtr  { nullptr };
static Fl_Image*  previewImagePtr{ nullptr };
static Fl_Text_Buffer* moreBufferPtr{ nullptr };
static Fl_Text_Editor* moreEditorPtr{ nullptr };

// 旧的路径
static std::string oldPath;
// 所有选择
static std::list<Fl_Choice*> choiceList;
// 图片列表
static std::vector<std::string> imageVector;
// 当前图片索引
static std::vector<std::string>::iterator imageIterator;

static void resetChoice(Fl_Widget*, void*);
static void loadImageVector(const std::string& path);
static void prevImage(Fl_Widget*, void*);
static void nextImage(Fl_Widget*, void*);
static void previewImage();

lifuren::ImageMarkWindow::ImageMarkWindow(int width, int height, const char* title) : MarkWindow(width, height, title) {
    this->imageMarkConfigPtr = &lifuren::config::CONFIG.imageMark;
}

lifuren::ImageMarkWindow::~ImageMarkWindow() {
    SPDLOG_DEBUG("关闭窗口：{}", __FILE__);
    lifuren::config::saveFile();
    // 清理数据
    oldPath = "";
    choiceList.clear();
    imageVector.clear();
    // 释放资源
    LFR_DELETE_PTR(prevPtr);
    LFR_DELETE_PTR(nextPtr);
    LFR_DELETE_PTR(resetPtr);
    LFR_DELETE_PTR(datasetPathPtr);
    LFR_DELETE_PTR(previewBoxPtr);
    LFR_DELETE_PTR(previewImagePtr);
    LFR_DELETE_PTR(moreEditorPtr);
    LFR_DELETE_PTR(moreBufferPtr);
}

void lifuren::ImageMarkWindow::drawElement() {
    // 配置按钮
    datasetPathPtr = new lifuren::Fl_Input_Directory_Chooser(110, 10, this->w() - 200, 30, "数据目录");
    // datasetPathPtr->value(this->imageMarkConfigPtr->datasetPath.c_str());
    prevPtr  = new Fl_Button(10,  50, 100, 30, "上张图片");
    nextPtr  = new Fl_Button(120, 50, 100, 30, "下张图片");
    resetPtr = new Fl_Button(230, 50, 100, 30, "重置选项");
    // LFR_INPUT_DIRECTORY_CHOOSER_CALLBACK(datasetPathPtr, imageMarkConfigPtr, datasetPath, ImageMarkWindow, loadImageVector);
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
    LFR_CHOICE_ADD_LIST(400, yPos, LABEL_IMAGE, xieziPtr , "衣着", "鞋子", "默认");
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
    // 更多
    yPos += 40;
    moreEditorPtr = new Fl_Text_Editor(10, yPos, 720, 100);
    moreBufferPtr = new Fl_Text_Buffer();
    moreEditorPtr->buffer(moreBufferPtr);
    moreEditorPtr->wrap_mode(moreEditorPtr->WRAP_AT_COLUMN, moreEditorPtr->textfont());
    moreEditorPtr->end();
    // 事件
    prevPtr->callback(prevImage, this);
    nextPtr->callback(nextImage, this);
    resetPtr->callback(resetChoice, this);
    // 加载资源
    // loadImageVector(this->imageMarkConfigPtr->datasetPath);
}

static void resetChoice(Fl_Widget*, void*) {
    for(auto ptr : choiceList) {
        const int index = ptr->find_index("默认");
        ptr->value(index);
    }
}

static void prevImage(Fl_Widget* widgetPtr, void* voidPtr) {
    if(imageVector.empty()) {
        SPDLOG_DEBUG("没有图片文件：{}", oldPath);
        return;
    }
    if(imageIterator == imageVector.begin()) {
        imageIterator = imageVector.end();
    }
    --imageIterator;
    previewImage();
}

static void nextImage(Fl_Widget* widgetPtr, void* voidPtr) {
    if(imageVector.empty()) {
        SPDLOG_DEBUG("没有图片文件：{}", oldPath);
        return;
    }
    ++imageIterator;
    if(imageIterator == imageVector.end()) {
        imageIterator = imageVector.begin();
    }
    previewImage();
}

static void loadImageVector(const std::string& path) {
    if(path.empty() || path == oldPath) {
        SPDLOG_DEBUG("忽略图片目录加载：{}", path);
        return;
    }
    oldPath = path;
    imageVector.clear();
    lifuren::files::listFiles(imageVector, oldPath, { ".jpg", ".jpeg", ".png" });
    imageIterator = imageVector.begin();
    previewImage();
}

static void previewImage() {
    if(imageIterator == imageVector.end()) {
        SPDLOG_WARN("没有可用图片");
        return;
    }
    SPDLOG_DEBUG("预览图片：{}", *imageIterator);
    // 释放资源
    LFR_DELETE_PTR(previewImagePtr);
    // 加载图片：异常处理
    Fl_Shared_Image* previewSharedPtr = Fl_Shared_Image::get((*imageIterator).c_str());
    const int boxWidth  = previewBoxPtr->w();
    const int boxHeight = previewBoxPtr->h();
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
    // 显示图片
    previewBoxPtr->image(previewImagePtr);
    previewBoxPtr->redraw();
}

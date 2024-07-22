#include "../../header/FLTK.hpp"

#include "spdlog/spdlog.h"

#include "utils/Files.hpp"
#include "config/Label.hpp"

#include "FL/Fl_Box.H"
#include "Fl/Fl_Shared_Image.H"

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
// 整体
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

// 旧的路径
static std::string oldPath;
// 图片列表
static std::vector<std::string> imageVector;
// 当前图片索引
static std::vector<std::string>::iterator imageIterator;

// 图片预览
static Fl_Box* previewBoxPtr = nullptr;
// 图片预览
static Fl_Image* previewImagePtr = nullptr;

/**
 * 加载图片
 * 
 * @param path 图片路径
 */
static void loadImageVector(const std::string& path);

// 上张图片
static void prevImage(Fl_Widget*, void*);
// 下张图片
static void nextImage(Fl_Widget*, void*);
// 预览图片
static void previewImage();

lifuren::ImageMarkWindow::ImageMarkWindow(int width, int height, const char* title) : MarkWindow(width, height, title) {
    this->loadConfig(lifuren::config::CONFIG_DATASET);
}

lifuren::ImageMarkWindow::~ImageMarkWindow() {
    SPDLOG_DEBUG("关闭窗口：{}", __FILE__);
    lifuren::config::saveFile(CONFIGS_PATH);
    // 清理数据
    oldPath = "";
    imageVector.clear();
    LFR_DELETE_PTR(previewBoxPtr);
    LFR_DELETE_PTR(previewImagePtr);
}

void lifuren::ImageMarkWindow::drawElement() {
    // 配置按钮
    this->datasetPathPtr = new Fl_Input_Directory_Chooser(100, 10, this->w() - 200, 30, "数据目录");
    this->datasetPathPtr->value(this->configPtr->datasetPath.c_str());
    this->prevPtr = new Fl_Button(10,  50, 100, 30, "上张图片");
    this->nextPtr = new Fl_Button(120, 50, 100, 30, "下张图片");
    this->prevPtr->callback(prevImage, this);
    this->nextPtr->callback(nextImage, this);
    LFR_INPUT_DIRECTORY_CHOOSER_CALLBACK(datasetPathPtr, datasetPath, ImageMarkWindow, loadImageVector);
    // 图片预览
    previewBoxPtr = new Fl_Box(this->w() / 2 + 200, this->h() / 2 - 150, 400, 300, "预览图片");
    previewBoxPtr->box(FL_FLAT_BOX);
    // 数学设置
    LFR_CHOICE_BUTTON(40,  90, LABEL_IMAGE, fasePtr,    "头部", "发色", "默认");
    LFR_CHOICE_BUTTON(160, 90, LABEL_IMAGE, faxingPtr,  "头部", "发型", "默认");
    LFR_CHOICE_BUTTON(280, 90, LABEL_IMAGE, meimaoPtr,  "头部", "眉毛", "默认");
    LFR_CHOICE_BUTTON(400, 90, LABEL_IMAGE, yanjingPtr, "头部", "眼睛", "默认");
    LFR_CHOICE_BUTTON(520, 90, LABEL_IMAGE, biziPtr,    "头部", "鼻子", "默认");
    LFR_CHOICE_BUTTON(640, 90, LABEL_IMAGE, yachiPtr,   "头部", "牙齿", "默认");
    // 加载资源
    loadImageVector(this->configPtr->datasetPath);
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

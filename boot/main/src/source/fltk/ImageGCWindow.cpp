#include "../../header/FLTK.hpp"

#include "utils/Jsons.hpp"

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
// 开始训练
static void trainStart(Fl_Widget*, void*);
// 结束训练
static void trainStop(Fl_Widget*, void*);
// 生成图片
static void generate(Fl_Widget*, void*);
// 预览图片
static void previewImage();

lifuren::ImageGCWindow::ImageGCWindow(int width, int height, const char* title) : ModelGCWindow(width, height, title) {
    this->loadConfig("ImageGC");
}

lifuren::ImageGCWindow::~ImageGCWindow() {
    SPDLOG_DEBUG("关闭窗口：{}", __FILE__);
    lifuren::config::saveFile(CONFIGS_PATH);
    // 清理数据
    oldPath = "";
    imageVector.clear();
    LFR_DELETE_PTR(previewBoxPtr);
    LFR_DELETE_PTR(previewImagePtr);
}

void lifuren::ImageGCWindow::drawElement() {
    this->modelPathPtr = new Fl_Input_Directory_Chooser(100, 10, this->w() - 200, 30, "模型目录");
    this->modelPathPtr->value(this->configPtr->modelPath.c_str());
    this->datasetPathPtr = new Fl_Input_Directory_Chooser(100, 50, this->w() - 200, 30, "数据目录");
    this->datasetPathPtr->value(this->configPtr->datasetPath.c_str());
    LFR_INPUT_DIRECTORY_CHOOSER(modelPathPtr, modelPath, ImageGCWindow);
    LFR_INPUT_DIRECTORY_CHOOSER_CALLBACK(datasetPathPtr, datasetPath, ImageGCWindow, loadImageVector);
    this->prevPtr       = new Fl_Button(10,  90, 100, 30, "上张图片");
    this->nextPtr       = new Fl_Button(120, 90, 100, 30, "下张图片");
    this->trainStartPtr = new Fl_Button(230, 90, 100, 30, "开始训练");
    this->trainStopPtr  = new Fl_Button(340, 90, 100, 30, "结束训练");
    this->generatePtr   = new Fl_Button(450, 90, 100, 30, "生成图片");
    this->prevPtr->callback(prevImage, this);
    this->nextPtr->callback(nextImage, this);
    this->trainStartPtr->callback(trainStart, this);
    this->trainStopPtr->callback(trainStop, this);
    this->generatePtr->callback(generate, this);
    // 图片预览
    previewBoxPtr = new Fl_Box(this->w() / 2 + 200, this->h() / 2 - 150, 400, 300, "预览图片");
    previewBoxPtr->box(FL_FLAT_BOX);
    // 设置：(10 + 110) * n + 40
    LFR_CHOICE_BUTTON(40,  130, LABEL_IMAGE, fasePtr,    "头部", "发色", "默认");
    LFR_CHOICE_BUTTON(160, 130, LABEL_IMAGE, faxingPtr,  "头部", "发型", "默认");
    LFR_CHOICE_BUTTON(280, 130, LABEL_IMAGE, meimaoPtr,  "头部", "眉毛", "默认");
    LFR_CHOICE_BUTTON(400, 130, LABEL_IMAGE, yanjingPtr, "头部", "眼睛", "默认");
    LFR_CHOICE_BUTTON(520, 130, LABEL_IMAGE, biziPtr,    "头部", "鼻子", "默认");
    LFR_CHOICE_BUTTON(640, 130, LABEL_IMAGE, yachiPtr,   "头部", "牙齿", "默认");
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

static void trainStart(Fl_Widget* widgetPtr, void* voidPtr) {
    const lifuren::ImageGCWindow* windowPtr = (lifuren::ImageGCWindow*) voidPtr;
}

static void trainStop(Fl_Widget* widgetPtr, void* voidPtr) {
    const lifuren::ImageGCWindow* windowPtr = (lifuren::ImageGCWindow*) voidPtr;
}

static void generate(Fl_Widget* widgetPtr, void* voidPtr) {
    const lifuren::ImageGCWindow* windowPtr = (lifuren::ImageGCWindow*) voidPtr;
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

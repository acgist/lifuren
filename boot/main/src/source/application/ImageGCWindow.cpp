#include "../../header/Window.hpp"

#include "Jsons.hpp"

// 旧的路径
static std::string oldPath;
// 图片列表
static std::vector<std::string> imageVector;
// 当前图片索引
static std::vector<std::string>::iterator iterator;

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

static void prevImage(Fl_Widget*, void*);
static void nextImage(Fl_Widget*, void*);
static void trainStart(Fl_Widget*, void*);
static void trainStop(Fl_Widget*, void*);
static void generate(Fl_Widget*, void*);
static void previewImage(Fl_Widget*, void*);

lifuren::ImageGCWindow::ImageGCWindow(int width, int height, const char* title) : ModelGCWindow(width, height, title) {
    auto iterator = SETTINGS.find("ImageGC");
    if(iterator == SETTINGS.end()) {
        this->settingPtr = new Setting();
        SETTINGS.insert(std::make_pair("ImageGC", *this->settingPtr));
    } else {
        this->settingPtr = &iterator->second;
    }
}

lifuren::ImageGCWindow::~ImageGCWindow() {
    SPDLOG_DEBUG("关闭ImageGCWindow");
    lifuren::jsons::saveFile(SETTINGS_PATH, lifuren::SETTINGS);
    LFR_DELETE_THIS_PTR(modelPathPtr);
    LFR_DELETE_THIS_PTR(datasetPathPtr);
    LFR_DELETE_THIS_PTR(prevPtr);
    LFR_DELETE_THIS_PTR(nextPtr);
    LFR_DELETE_THIS_PTR(trainStartPtr);
    LFR_DELETE_THIS_PTR(trainStopPtr);
    LFR_DELETE_THIS_PTR(generatePtr);
    // 清理数据
    oldPath = "";
    imageVector.clear();
    LFR_DELETE_PTR(previewBoxPtr);
    LFR_DELETE_PTR(previewImagePtr);
}

void lifuren::ImageGCWindow::drawElement() {
    this->modelPathPtr = new Fl_Input_Directory_Chooser(100, 10, this->w() - 200, 30, "模型目录");
    this->modelPathPtr->value(this->settingPtr->modelPath.c_str());
    this->datasetPathPtr = new Fl_Input_Directory_Chooser(100, 50, this->w() - 200, 30, "数据目录");
    this->datasetPathPtr->value(this->settingPtr->datasetPath.c_str());
    LFR_INPUT_DIRECTORY_CHOOSER_CALLBACK(modelPathPtr, modelPath, ImageGCWindow);
    LFR_INPUT_DIRECTORY_CHOOSER_CALLBACK(datasetPathPtr, datasetPath, ImageGCWindow);
    this->prevPtr = new Fl_Button(10,  90, 100, 30, "上一张图");
    this->nextPtr = new Fl_Button(120, 90, 100, 30, "下一张图");
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
    LFR_CHOICE_BUTTON(40,  130, fasePtr,   "头部", "发色", "默认");
    LFR_CHOICE_BUTTON(160, 130, faxingPtr, "头部", "发型", "默认");
}

static void prevImage(Fl_Widget* widgetPtr, void* voidPtr) {
    lifuren::ImageGCWindow* windowPtr = (lifuren::ImageGCWindow*) voidPtr;
    loadImageVector(windowPtr->datasetPath());
    if(imageVector.empty()) {
        return;
    }
    if(iterator == imageVector.begin()) {
        iterator = imageVector.end();
    }
    --iterator;
    previewImage(widgetPtr, voidPtr);
}

static void nextImage(Fl_Widget* widgetPtr, void* voidPtr) {
    lifuren::ImageGCWindow* windowPtr = (lifuren::ImageGCWindow*) voidPtr;
    loadImageVector(windowPtr->datasetPath());
    if(imageVector.empty()) {
        return;
    }
    ++iterator;
    if(iterator == imageVector.end()) {
        iterator = imageVector.begin();
    }
    previewImage(widgetPtr, voidPtr);
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
    if(path.empty()) {
        SPDLOG_DEBUG("目录无效：{} - {}", __func__, path);
        return;
    }
    if(path == oldPath) {
        SPDLOG_DEBUG("目录没有改变：{} - {}", __func__, path);
        return;
    }
    oldPath = path;
    imageVector.clear();
    lifuren::files::listFiles(imageVector, oldPath, { ".jpg", ".jpeg", ".png" });
    iterator = imageVector.begin();
}

static void previewImage(Fl_Widget* widgetPtr, void* voidPtr) {
    SPDLOG_DEBUG("预览图片：{} - {}", __func__, *iterator);
    // 释放资源
    LFR_DELETE_PTR(previewImagePtr);
    // 加载图片：异常处理
    Fl_Shared_Image* previewSharedPtr = Fl_Shared_Image::get((*iterator).c_str());
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

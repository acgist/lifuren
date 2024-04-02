/**
 * FLTK GUI
 * 
 * @author acgist
 */
#pragma once

#include <string>
#include <algorithm>

#include "FL/Fl.H"
#include "FL/fl_ask.H"
#include "FL/Fl_Box.H"
#include "FL/Fl_Input.H"
#include "FL/Fl_Button.H"
#include "FL/Fl_Choice.H"
#include "FL/Fl_Window.H"
#include "FL/Fl_PNG_Image.H"
#include "FL/Fl_JPEG_Image.H"
#include "FL/Fl_Text_Buffer.H"
#include "FL/Fl_Text_Display.H"
#include "Fl/Fl_Shared_Image.H"
#include "Fl/Fl_Native_File_Chooser.H"

#include "Ptr.hpp"
#include "Logger.hpp"
#include "utils/Files.hpp"
#include "config/Label.hpp"
#include "config/Config.hpp"

// 模型变量模型
#ifndef LFR_MODEL_DEFINE
#define LFR_MODEL_DEFINE(modelTypeLower, modelTypeUpper)             \
    Fl_Button* modelTypeLower##GcPtr = nullptr;                      \
    Fl_Button* modelTypeLower##TsPtr = nullptr;                      \
    modelTypeUpper##GCWindow* modelTypeLower##GcWindowPtr = nullptr; \
    modelTypeUpper##TSWindow* modelTypeLower##TsWindowPtr = nullptr;
#endif

// 目录选择
#ifndef LFR_INPUT_DIRECTORY_CHOOSER
#define LFR_INPUT_DIRECTORY_CHOOSER(inputPtr, configName, windowName)   \
    this->inputPtr->callback([](Fl_Widget* widgetPtr, void* voidPtr) {  \
        const char* value = ((windowName*) voidPtr)->inputPtr->value(); \
        if(value == nullptr || strlen(value) == 0) {                    \
            return;                                                     \
        }                                                               \
        ((windowName*) voidPtr)->configPtr->configName = value;         \
    }, this);
#endif

// 目录选择回调
#ifndef LFR_INPUT_DIRECTORY_CHOOSER_CALLBACK
#define LFR_INPUT_DIRECTORY_CHOOSER_CALLBACK(inputPtr, configName, windowName, callbackName) \
    this->inputPtr->callback([](Fl_Widget* widgetPtr, void* voidPtr) {                       \
        const char* value = ((windowName*) voidPtr)->inputPtr->value();                      \
        if(value == nullptr || strlen(value) == 0) {                                         \
            return;                                                                          \
        }                                                                                    \
        ((windowName*) voidPtr)->configPtr->configName = value;                              \
        callbackName(value);                                                                 \
    }, this);
#endif

// 下拉选择
#ifndef LFR_CHOICE_BUTTON
#define LFR_CHOICE_BUTTON(x, y, map, buttonPtr, groupName, labelName, defaultValue)           \
{                                                                                             \
    this->buttonPtr = new Fl_Choice(x, y, 80, 30, labelName);                                 \
    this->buttonPtr->add(defaultValue);                                                       \
    auto iterator = lifuren::map.find(groupName);                                             \
    if(iterator != lifuren::map.end()) {                                                      \
        std::vector<LabelFile> vector = iterator->second;                                     \
        std::for_each(vector.begin(), vector.end(), [this](auto& label) {                     \
            if(label.name == labelName) {                                                     \
                std::for_each(label.labels.begin(), label.labels.end(), [this](auto& value) { \
                    this->buttonPtr->add(value.c_str());                                      \
                });                                                                           \
            }                                                                                 \
        });                                                                                   \
    }                                                                                         \
    auto defaultPtr = this->buttonPtr->find_item(defaultValue);                               \
    this->buttonPtr->value(defaultPtr);                                                       \
}
#endif

// 图片缩放
#ifndef LFR_IMAGE_PREVIEW_SCALE
#define LFR_IMAGE_PREVIEW_SCALE 1.2
#endif

namespace lifuren {

// 加载FLTK窗口
extern void initFltkWindow();

// 关闭FLTK窗口
extern void shutdownFltkWindow();

/**
 * 抽象窗口
 */
class Window : public Fl_Window {

protected:
    /**
     * 图标指针
     */
    Fl_PNG_Image* iconImagePtr = nullptr;

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    Window(int width, int height, const char* title);
    /**
     * 析构函数
     */
    virtual ~Window();

public:
    /**
     * 加载窗口
     */
    virtual void init();

protected:
    /**
     * 加载组件
     */
    virtual void drawElement() = 0;
    /**
     * 设置图标
     */
    void icon();
    /**
     * 窗口居中
     */
    void center();

};

// 定义窗口

class AudioGCWindow;
class AudioTSWindow;
class ImageGCWindow;
class ImageTSWindow;
class VideoGCWindow;
class VideoTSWindow;
class PoetryGCWindow;
class PoetryTSWindow;
class AboutWindow;

/**
 * 主窗口
 */
class MainWindow : public Window {

private:
    // 音频模块
    LFR_MODEL_DEFINE(audio,  Audio);
    // 图片模块
    LFR_MODEL_DEFINE(image,  Image);
    // 视频模块
    LFR_MODEL_DEFINE(video,  Video);
    // 诗词模块
    LFR_MODEL_DEFINE(poetry, Poetry);
    // 关于按钮
    Fl_Button* aboutButtonPtr = nullptr;
    // 重新加载配置按钮
    Fl_Button* reloadButtonPtr = nullptr;
    // 关于窗口
    AboutWindow* aboutWindowPtr = nullptr;

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    MainWindow(int width, int height, const char* title);
    /**
     * 析构函数
     */
    virtual ~MainWindow();

public:
    // 音频风格迁移
    void audioTs();
    // 图片内容生成
    void imageGc();
    // 图片风格迁移
    void imageTs();
    // 视频内容生成
    void videoGc();
    // 视频风格迁移
    void videoTs();
    // 诗词内容生成
    void poetryGc();
    // 关于
    void about();

protected:
    /**
     * 加载组件
     */
    virtual void drawElement() override;

};

/**
 * 关于窗口
 */
class AboutWindow : public Window {

private:
    // 官网
    Fl_Button* homePagePtr = nullptr;
    // 关于内容
    Fl_Text_Buffer* aboutBufferPtr = nullptr;
    // 关于组件
    Fl_Text_Display* aboutDisplayPtr = nullptr;

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    AboutWindow(int width, int height, const char* title = "关于");
    /**
     * 析构函数
     */
    virtual ~AboutWindow();

protected:
    /**
     * 加载组件
     */
    virtual void drawElement() override;

};

/**
 * 图片标记
 * 
 * config/image.yml
 */
class ImageLabel {

protected:
    // 头部
    Fl_Choice* fasePtr     = nullptr;
    Fl_Choice* faxingPtr   = nullptr;
    Fl_Choice* meimaoPtr   = nullptr;
    Fl_Choice* yanjingPtr  = nullptr;
    Fl_Choice* biziPtr     = nullptr;
    Fl_Choice* yachiPtr    = nullptr;
    Fl_Choice* zuibaPtr    = nullptr;
    Fl_Choice* kouhongPtr  = nullptr;
    Fl_Choice* biaoqingPtr = nullptr;
    Fl_Choice* lianxingPtr = nullptr;
    // 上身
    Fl_Choice* rufangPtr   = nullptr;
    Fl_Choice* shouxingPtr = nullptr;
    Fl_Choice* yaobuPtr    = nullptr;
    // 下身
    Fl_Choice* tunbuPtr   = nullptr;
    Fl_Choice* tuixingPtr = nullptr;
    // 衣着
    Fl_Choice* sediaoPtr = nullptr;
    Fl_Choice* yifuPtr   = nullptr;
    Fl_Choice* kuziPtr   = nullptr;
    Fl_Choice* xieziPtr  = nullptr;
    // 饰品
    Fl_Choice* toushiPtr  = nullptr;
    Fl_Choice* ershiPtr   = nullptr;
    Fl_Choice* yanshiPtr  = nullptr;
    Fl_Choice* lianshiPtr = nullptr;
    Fl_Choice* shoushiPtr = nullptr;
    Fl_Choice* baobaoPtr  = nullptr;
    // 整体
    Fl_Choice* fusePtr     = nullptr;
    Fl_Choice* pifuPtr     = nullptr;
    Fl_Choice* xinggePtr   = nullptr;
    Fl_Choice* nianlingPtr = nullptr;
    Fl_Choice* shengaoPtr  = nullptr;
    Fl_Choice* tixingPtr   = nullptr;
    Fl_Choice* titaiPtr    = nullptr;
    Fl_Choice* zhiyePtr    = nullptr;
    // 环境
    Fl_Choice* tianqiPtr   = nullptr;
    Fl_Choice* qianjingPtr = nullptr;
    Fl_Choice* beijingPtr  = nullptr;

};

/**
 * 模型窗口
 */
class ModelWindow : public Window {

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    ModelWindow(int width, int height, const char* title);
    /**
     * 析构函数
     */
    virtual ~ModelWindow();

protected:
    // 全局配置：不用释放
    Config* configPtr = nullptr;
    // 模型路径
    Fl_Input* modelPathPtr = nullptr;
    // 数据路径
    Fl_Input* datasetPathPtr = nullptr;

protected:
    /**
     * @param modelType 模型类型
     */
    void loadConfig(const std::string& modelType);

};

/**
 * 内容生成窗口
 * 
 * ModelGCWindow
 */
class ModelGCWindow : public ModelWindow {

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    ModelGCWindow(int width, int height, const char* title);
    /**
     * 析构函数
     */
    virtual ~ModelGCWindow();

protected:
    // 上个内容
    Fl_Button* prevPtr = nullptr;
    // 下个内容
    Fl_Button* nextPtr = nullptr;
    // 开始训练
    Fl_Button* trainStartPtr = nullptr;
    // 结束训练
    Fl_Button* trainStopPtr  = nullptr;
    // 内容生成
    Fl_Button* generatePtr   = nullptr;

};

/**
 * 风格迁移窗口
 * 
 * ModelTSWindow
 */
class ModelTSWindow : public ModelWindow {

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    ModelTSWindow(int width, int height, const char* title);
    /**
     * 析构函数
     */
    virtual ~ModelTSWindow();

protected:
    // 开始训练
    Fl_Button* trainStartPtr = nullptr;
    // 结束训练
    Fl_Button* trainStopPtr  = nullptr;
    // 风格迁移
    Fl_Button* transferPtr   = nullptr;

};

/**
 * 音频内容生成
 * 
 * @see AudioGC
 * 
* @deprecated 不会实现
 */
class AudioGCWindow : public ModelGCWindow {
};

/**
 * 音频风格迁移
 * 
 * @see AudioTS
 */
class AudioTSWindow : public ModelTSWindow {

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    AudioTSWindow(int width, int height, const char* title = "音频风格迁移");
    /**
     * 析构函数
     */
    virtual ~AudioTSWindow();

protected:
    /**
     * 加载组件
     */
    virtual void drawElement() override;

};

/**
 * 图片内容生成
 * 
 * @see ImageGC
 */
class ImageGCWindow : public ModelGCWindow, public ImageLabel {

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    ImageGCWindow(int width, int height, const char* title = "图片内容生成");
    /**
     * 析构函数
     */
    virtual ~ImageGCWindow();

protected:
    /**
     * 加载组件
     */
    virtual void drawElement() override;

};

/**
 * 图片风格迁移
 * 
 * @see ImageTS
 */
class ImageTSWindow : public ModelTSWindow {

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    ImageTSWindow(int width, int height, const char* title = "图片风格迁移");
    /**
     * 析构函数
     */
    virtual ~ImageTSWindow();

protected:
    /**
     * 加载组件
     */
    virtual void drawElement() override;

};

/**
 * 诗词内容生成
 * 
 * @see PoetryGC
 */
class PoetryGCWindow : public ModelGCWindow {

public:
    // 自动标记：通过已有标记自动标记
    Fl_Button* autoMarkPtr = nullptr;

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    PoetryGCWindow(int width, int height, const char* title = "诗词内容生成");
    /**
     * 析构函数
     */
    virtual ~PoetryGCWindow();

protected:
    /**
     * 加载组件
     */
    virtual void drawElement() override;

};

/**
 * 诗词风格迁移
 * 
 * @see PoetryTS
 * 
 * @deprecated 不会实现
 */
class PoetryTSWindow : public ModelTSWindow {
};

/**
 * 视频内容生成
 * 
 * @see VideoGC
 */
class VideoGCWindow : public ModelGCWindow {

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    VideoGCWindow(int width, int height, const char* title = "视频内容生成");
    /**
     * 析构函数
     */
    virtual ~VideoGCWindow();

protected:
    /**
     * 加载组件
     */
    virtual void drawElement() override;

};

/**
 * 视频风格迁移
 * 
 * @see VideoTS
 */
class VideoTSWindow : public ModelTSWindow {

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    VideoTSWindow(int width, int height, const char* title = "视频风格迁移");
    /**
     * 析构函数
     */
    virtual ~VideoTSWindow();

protected:
    /**
     * 加载组件
     */
    virtual void drawElement() override;

};

/**
 * 目录选择组件
 */
class Fl_Input_Directory_Chooser : public Fl_Input {

private:
    // 标题
    const char* title;
    // 目录
    const char* directory = ".";

public:
    /**
     * @param x         x
     * @param y         y
     * @param width     宽度
     * @param height    高度
     * @param title     标题
     * @param directory 当前目录
     */
    Fl_Input_Directory_Chooser(
        int x,
        int y,
        int width,
        int height,
        const char* title,
        const char* directory = "."
    );
    /**
     * 析构函数
     */
    virtual ~Fl_Input_Directory_Chooser();

public:
    /**
     * @param event 事件
     */
    int handle(int event) override;

};

/**
 * @param title     标题
 * @param directory 当前目录
 * @param filter    文件过滤（*.{cxx,cpp}）
 * 
 * @return 选择文件路径
 */
extern std::string fileChooser(const char* title, const char* directory = ".", const char* filter = "*.*");

/**
 * @param title     标题
 * @param directory 当前目录
 * 
 * @return 选择目录路径
 */
extern std::string directoryChooser(const char* title, const char* directory = ".");

} // END lifuren

/**
 * FLTK GUI
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_BOOT_FLTK_HPP
#define LFR_HEADER_BOOT_FLTK_HPP

#include <string>
#include <cstring>
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
#include "config/Label.hpp"
#include "config/Config.hpp"

// 半个窗口宽度
#ifndef LFR_HALF_WIDTH
#define LFR_HALF_WIDTH (this->w() - 60) / 2
#endif

// 模型变量
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
        if(value == nullptr || std::strlen(value) == 0) {               \
            return;                                                     \
        }                                                               \
        ((windowName*) voidPtr)->configPtr->configName = value;         \
    }, this);
#endif

// 目录选择并且回调
#ifndef LFR_INPUT_DIRECTORY_CHOOSER_CALLBACK
#define LFR_INPUT_DIRECTORY_CHOOSER_CALLBACK(inputPtr, configName, windowName, callbackName) \
    this->inputPtr->callback([](Fl_Widget* widgetPtr, void* voidPtr) {                       \
        const char* value = ((windowName*) voidPtr)->inputPtr->value();                      \
        if(value == nullptr || std::strlen(value) == 0) {                                    \
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
    buttonPtr = new Fl_Choice(x, y, 80, 30, labelName);                                 \
    buttonPtr->add(defaultValue);                                                       \
    auto iterator = lifuren::map.find(groupName);                                             \
    if(iterator != lifuren::map.end()) {                                                      \
        std::vector<LabelFile>& vector = iterator->second;                                    \
        std::for_each(vector.begin(), vector.end(), [this](auto& label) {                     \
            if(label.name == labelName) {                                                     \
                std::for_each(label.labels.begin(), label.labels.end(), [this](auto& value) { \
                    buttonPtr->add(value.c_str());                                      \
                });                                                                           \
            }                                                                                 \
        });                                                                                   \
    }                                                                                         \
    auto defaultPtr = buttonPtr->find_item(defaultValue);                               \
    buttonPtr->value(defaultPtr);                                                       \
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
    // 全局配置：不用释放
    Config* configPtr = nullptr;
    // 图标指针
    Fl_PNG_Image* iconImagePtr = nullptr;

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    Window(int width, int height, const char* title);
    // 析构函数
    virtual ~Window();

public:
    // 加载窗口
    virtual void init();

protected:
    // 加载组件
    virtual void drawElement() = 0;
    // 设置图标
    void icon();
    // 窗口居中
    void center();
    /**
     * 加载配置
     * 
     * @param modelType 模型类型
     */
    void loadConfig(const std::string& modelType);

};

// 定义所有窗口
class AudioGCWindow;
class AudioTSWindow;
class ImageGCWindow;
class ImageTSWindow;
class VideoGCWindow;
class VideoTSWindow;
class PoetryGCWindow;
class PoetryTSWindow;
class ImageMarkWindow;
class PoetryMarkWindow;
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
    // 图片标记按钮
    Fl_Button* imageMarkButtonPtr = nullptr;
    // 图片标记窗口
    ImageMarkWindow* imageMarkWindowPtr = nullptr;
    // 诗词标记按钮
    Fl_Button* poetryMarkButtonPtr = nullptr;
    // 诗词标记窗口
    PoetryMarkWindow* poetryMarkWindowPtr = nullptr;
    // 关于按钮
    Fl_Button* aboutButtonPtr = nullptr;
    // 关于窗口
    AboutWindow* aboutWindowPtr = nullptr;
    // 重新加载配置按钮
    Fl_Button* reloadButtonPtr = nullptr;

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    MainWindow(int width, int height, const char* title);
    // 析构函数
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
    // 图片标记
    void imageMark();
    // 诗词标记
    void poetryMark();
    // 关于
    void about();

protected:
    // 加载组件
    virtual void drawElement() override;

};

/**
 * 标记窗口
 */
class MarkWindow : public Window {

public:
    // 上个内容
    Fl_Button* prevPtr = nullptr;
    // 下个内容
    Fl_Button* nextPtr = nullptr;
    // 数据路径
    Fl_Input* datasetPathPtr = nullptr;

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    MarkWindow(int width, int height, const char* title = "关于");
    // 析构函数
    virtual ~MarkWindow();

};

/**
 * 图片标记窗口
 */
class ImageMarkWindow : public MarkWindow {

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    ImageMarkWindow(int width, int height, const char* title = "图片标记");
    // 析构函数
    virtual ~ImageMarkWindow();

protected:
    // 加载组件
    virtual void drawElement() override;

};

/**
 * 诗词标记窗口
 */
class PoetryMarkWindow : public MarkWindow {

public:
    // 自动标记：通过已有标记自动标记
    Fl_Button* autoMarkPtr = nullptr;

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    PoetryMarkWindow(int width, int height, const char* title = "诗词标记");
    // 析构函数
    virtual ~PoetryMarkWindow();

protected:
    // 加载组件
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
    // 析构函数
    virtual ~AboutWindow();

protected:
    // 加载组件
    virtual void drawElement() override;

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
    // 析构函数
    virtual ~ModelWindow();

protected:
    // 模型路径
    Fl_Input* modelPathPtr = nullptr;
    // 数据路径
    Fl_Input* datasetPathPtr = nullptr;
    // 开始训练
    Fl_Button* trainStartPtr = nullptr;
    // 结束训练
    Fl_Button* trainStopPtr  = nullptr;
    // TODO: 微调
    // TODO: 量化

};

/**
 * 内容生成窗口
 */
class ModelGCWindow : public ModelWindow {

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    ModelGCWindow(int width, int height, const char* title);
    // 析构函数
    virtual ~ModelGCWindow();

protected:
    // 内容生成
    Fl_Button* generatePtr = nullptr;

};

/**
 * 风格迁移窗口
 */
class ModelTSWindow : public ModelWindow {

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    ModelTSWindow(int width, int height, const char* title);
    // 析构函数
    virtual ~ModelTSWindow();

protected:
    // 风格迁移
    Fl_Button* transferPtr = nullptr;

};

/**
 * 音频内容生成
 * 
 * @see AudioGC
 * 
 * @deprecated
 */
class AudioGCWindow : public ModelGCWindow {

public:
    // 不会实现
    AudioGCWindow() = delete;

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
    // 析构函数
    virtual ~AudioTSWindow();

protected:
    // 加载组件
    virtual void drawElement() override;

};

/**
 * 图片内容生成
 * 
 * @see ImageGC
 */
class ImageGCWindow : public ModelGCWindow {

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    ImageGCWindow(int width, int height, const char* title = "图片内容生成");
    // 析构函数
    virtual ~ImageGCWindow();

protected:
    // 加载组件
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
    // 析构函数
    virtual ~ImageTSWindow();

protected:
    // 加载组件
    virtual void drawElement() override;

};

/**
 * 诗词内容生成
 * 
 * @see PoetryGC
 */
class PoetryGCWindow : public ModelGCWindow {

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    PoetryGCWindow(int width, int height, const char* title = "诗词内容生成");
    // 析构函数
    virtual ~PoetryGCWindow();

protected:
    // 加载组件
    virtual void drawElement() override;

};

/**
 * 诗词风格迁移
 * 
 * @see PoetryTS
 * 
 * @deprecated
 */
class PoetryTSWindow : public ModelTSWindow {

public:
    // 不会实现
    PoetryTSWindow() = delete;

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
    // 析构函数
    virtual ~VideoGCWindow();

protected:
    // 加载组件
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
    // 析构函数
    virtual ~VideoTSWindow();

protected:
    // 加载组件
    virtual void drawElement() override;

};

/**
 * 目录选择组件
 */
class Fl_Input_Directory_Chooser : public Fl_Input {

private:
    // 标题
    const char* title = nullptr;
    // 目录
    const char* directory = nullptr;

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
    // 析构函数
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
 * @param filter    文件过滤：*.{cxx,cpp}
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

#endif // LFR_HEADER_BOOT_FLTK_HPP

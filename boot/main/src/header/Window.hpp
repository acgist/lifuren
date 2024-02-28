/**
 * 窗口定义
 * 
 * @author acgist
 */
#pragma once

#include <string>

#include "Ptr.hpp"
#include "Logger.hpp"
#include "Setting.hpp"

#include "FL/Fl.H"
#include "FL/fl_ask.H"
#include "FL/Fl_Input.H"
#include "FL/Fl_Button.H"
#include "FL/Fl_Window.H"
#include "FL/Fl_PNG_Image.H"
#include "FL/Fl_Text_Buffer.H"
#include "FL/Fl_Text_Display.H"
#include "Fl/Fl_Shared_Image.H"
#include "Fl/Fl_Native_File_Chooser.H"

#ifndef LFR_MODEL_MODULE
#define LFR_MODEL_MODULE(modelTypeLower, modelTypeUpper)             \
    Fl_Button* modelTypeLower##GcPtr = nullptr;                      \
    Fl_Button* modelTypeLower##TsPtr = nullptr;                      \
    modelTypeUpper##GCWindow* modelTypeLower##GcWindowPtr = nullptr; \
    modelTypeUpper##GCWindow* modelTypeLower##TsWindowPtr = nullptr;
#endif

#ifndef LFR_INPUT_DIRECTORY_CHOOSER_CALLBACK
#define LFR_INPUT_DIRECTORY_CHOOSER_CALLBACK(inputPtr, settingName, windowName)                        \
    this->inputPtr->callback([](Fl_Widget* widgetPtr, void* voidPtr) {                                 \
        ((windowName*) voidPtr)->settingPtr->settingName = ((windowName*) voidPtr)->inputPtr->value(); \
    }, this);
#endif

namespace lifuren {

/**
 * 抽象窗口
 */
class LFRWindow : public Fl_Window {

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
    LFRWindow(int width, int height, const char* title);
    virtual ~LFRWindow();

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
class MainWindow : public LFRWindow {

private:
    LFR_MODEL_MODULE(audio, Audio);
    LFR_MODEL_MODULE(image, Image);
    LFR_MODEL_MODULE(video, Video);
    LFR_MODEL_MODULE(poetry, Poetry);
    // 关于按钮
    Fl_Button* aboutButtonPtr = nullptr;
    // 关于窗口
    AboutWindow* aboutWindowPtr = nullptr;

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    MainWindow(int width, int height, const char* title);
    virtual ~MainWindow();

public:
    // ImageGC
    void imageGc();
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
class AboutWindow : public LFRWindow {

private:
    // 官网
    Fl_Button* homePagePtr = nullptr;
    // 关于内容
    Fl_Text_Buffer* aboutBufferPtr = nullptr;
    // 关于
    Fl_Text_Display* aboutDisplayPtr = nullptr;

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    AboutWindow(int width, int height, const char* title = "关于");
    virtual ~AboutWindow();

protected:
    /**
     * 加载组件
     */
    virtual void drawElement() override;

};

/**
 * 模型窗口
 */
class ModelWindow : public LFRWindow {

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    ModelWindow(int width, int height, const char* title);
    virtual ~ModelWindow();

protected:
    // 配置：不用释放
    Setting* settingPtr = nullptr;
    // 模型路径
    Fl_Input* modelPathPtr = nullptr;
    // 数据路径
    Fl_Input* datasetPathPtr = nullptr;

};

/**
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
    Fl_Button* generatePtr = nullptr;

};

/**
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
    virtual ~ModelTSWindow();

protected:
    // 开始训练
    Fl_Button* trainStartPtr = nullptr;
    // 结束训练
    Fl_Button* trainStopPtr  = nullptr;
    // 输入文件
    Fl_Input* sourceFilePtr = nullptr;
    // 风格迁移
    Fl_Button* transferPtr  = nullptr;

};

/**
 * @see AudioGC
 * 
* @deprecated 不会实现
 */
class AudioGCWindow : public ModelGCWindow {
};

/**
 * @see AudioTS
 */
class AudioTSWindow : public ModelTSWindow {
};

/**
 * @see ImageGC
 */
class ImageGCWindow : public ModelGCWindow {

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    ImageGCWindow(int width, int height, const char* title = "图片生成");
    virtual ~ImageGCWindow();

protected:
    /**
     * 加载组件
     */
    virtual void drawElement() override;

};

/**
 * @see ImageTS
 */
class ImageTSWindow : public ModelTSWindow {
};

/**
 * @see PoetryGC
 */
class PoetryGCWindow : public ModelGCWindow {

private:
    // 自动标记：通过已有标记自动标记
    Fl_Button* autoMarkPtr = nullptr;

};

/**
 * @see PoetryTS
 * 
 * @deprecated 不会实现
 */
class PoetryTSWindow : public ModelTSWindow {
};

/**
 * @see VideoGC
 */
class VideoGCWindow : public ModelGCWindow {
};

/**
 * @see VideoTS
 */
class VideoTSWindow : public ModelTSWindow {
};

/**
 * 路径选择Input
 */
class Fl_Input_Directory_Chooser : public Fl_Input {

private:
    // 标题
    const char* title;
    // 目录
    const char* directory = ".";

public:
    Fl_Input_Directory_Chooser(
        int x,
        int y,
        int width,
        int height,
        const char* title,
        const char* directory = "."
    );
    ~Fl_Input_Directory_Chooser();

public:
    int handle(int event) override;

};

/**
 * @param title     标题
 * @param directory 选择目录
 * @param filter    文件过滤（*.{cxx,cpp}）
 * 
 * @return 选择文件路径
 */
extern std::string fileChooser(const char* title, const char* directory = ".", const char* filter = "*.*");

/**
 * @param title     标题
 * @param directory 选择目录
 * 
 * @return 选择目录路径
 */
extern std::string directoryChooser(const char* title, const char* directory = ".");

}

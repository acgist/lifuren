/**
 * 窗口定义
 * 
 * @author acgist
 */
#pragma once

#include "Ptr.hpp"
#include "Logger.hpp"

#include "FL/Fl.H"
#include "FL/Fl_Button.H"
#include "FL/Fl_Window.H"
#include "FL/Fl_PNG_Image.H"
#include "FL/Fl_Text_Buffer.H"
#include "FL/Fl_Text_Display.H"

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
     * @param width    窗口宽度
     * @param height   窗口高度
     * @param titlePtr 窗口名称
     */
    LFRWindow(int width, int height, const char* titlePtr);
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
    // AudioGC按钮
    Fl_Button* audioGcPtr = nullptr;
    // AudioTS按钮
    Fl_Button* audioTsPtr = nullptr;
    // AudioGC窗口
    AudioGCWindow* audioGcWindowPtr = nullptr;
    // AudioTS窗口
    AudioGCWindow* audioTsWindowPtr = nullptr;
    // Audio分组
    Fl_Group* audioGroupPtr = nullptr;
    // ImageGC按钮
    Fl_Button* imageGcPtr = nullptr;
    // ImageTS按钮
    Fl_Button* imageTsPtr = nullptr;
    // ImageGC窗口
    ImageGCWindow* imageGcWindowPtr = nullptr;
    // ImageTS窗口
    ImageGCWindow* imageTsWindowPtr = nullptr;
    // Image分组
    Fl_Group* imageGroupPtr = nullptr;
    // VideoGC按钮
    Fl_Button* videoGcPtr = nullptr;
    // VideoTS按钮
    Fl_Button* videoTsPtr = nullptr;
    // VideoGC窗口
    VideoGCWindow* videoGcWindowPtr = nullptr;
    // VideoTS窗口
    VideoGCWindow* videoTsWindowPtr = nullptr;
    // Video分组
    Fl_Group* videoGroupPtr = nullptr;
    // PoetryGC按钮
    Fl_Button* poetryGcPtr = nullptr;
    // PoetryTS按钮
    Fl_Button* poetryTsPtr = nullptr;
    // PoetryGC窗口
    PoetryGCWindow* poetryGcWindowPtr = nullptr;
    // PoetryTS窗口
    PoetryTSWindow* poetryTsWindowPtr = nullptr;
    // Poetry分组
    Fl_Group* poetryGroupPtr = nullptr;
    // 关于按钮
    Fl_Button* aboutButtonPtr = nullptr;
    // 关于窗口
    AboutWindow* aboutWindowPtr = nullptr;

public:
    /**
     * @param width    窗口宽度
     * @param height   窗口高度
     * @param titlePtr 窗口名称
     */
    MainWindow(int width, int height, const char* titlePtr);
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
     * @param width    窗口宽度
     * @param height   窗口高度
     * @param titlePtr 窗口名称
     */
    AboutWindow(int width, int height, const char* titlePtr);
    virtual ~AboutWindow();

protected:
    /**
     * 加载组件
     */
    virtual void drawElement() override;

};

/**
 * @see AudioGC
 */
class AudioGCWindow : public LFRWindow {

};

/**
 * @see AudioTS
 */
class AudioTSWindow : public LFRWindow {

};

/**
 * @see ImageGC
 */
class ImageGCWindow : public LFRWindow {

public:
    /**
     * @param width    窗口宽度
     * @param height   窗口高度
     * @param titlePtr 窗口名称
     */
    ImageGCWindow(int width, int height, const char* titlePtr);
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
class ImageTSWindow : public LFRWindow {

};

/**
 * @see PoetryGC
 */
class PoetryGCWindow : public LFRWindow {

};

/**
 * @see PoetryTS
 */
class PoetryTSWindow : public LFRWindow {

};

/**
 * @see VideoGC
 */
class VideoGCWindow : public LFRWindow {

};

/**
 * @see VideoTS
 */
class VideoTSWindow : public LFRWindow {

};

}

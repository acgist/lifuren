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

#ifndef MEDIA_MODULE
#define MEDIA_MODULE(mediaTypeLower, mediaTypeUpper)                     \
    Fl_Button* mediaTypeLower##GcPtr = nullptr;                        \
    Fl_Button* mediaTypeLower##TsPtr = nullptr;                        \
    mediaTypeUpper##GCWindow* mediaTypeLower##GcWindowPtr = nullptr; \
    mediaTypeUpper##GCWindow* mediaTypeLower##TsWindowPtr = nullptr; \
    Fl_Group* mediaTypeLower##GroupPtr = nullptr;
#endif

/**
 * 主窗口
 */
class MainWindow : public LFRWindow {

private:
    MEDIA_MODULE(audio, Audio);
    MEDIA_MODULE(image, Image);
    MEDIA_MODULE(video, Video);
    MEDIA_MODULE(poetry, Poetry);
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

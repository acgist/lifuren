/**
 * 窗口定义
 * 
 * @author acgist
 */
#pragma once

#include "GLog.hpp"

#include "FL/Fl.H"
#include "FL/Fl_Input.H"
#include "FL/Fl_Button.H"
#include "FL/Fl_Window.H"
#include "FL/Fl_PNG_Image.H"
#include "FL/Fl_JPEG_Image.H"

namespace lifuren {

/**
 * 抽象窗口
 */
class LFRWindow : public Fl_Window {

public:
    /**
     * @param width    窗口宽度
     * @param height   窗口高度
     * @param titlePtr 窗口名称
     */
    LFRWindow(int width, int height, const char* titlePtr);
    virtual ~LFRWindow();

protected:
    /**
     * 加载窗口
     */
    virtual void init();
    /**
     * 设置图标
     */
    void icon();
    /**
     * 窗口居中
     */
    void center();

};

/**
 * 主窗口
 */
class MainWindow : public LFRWindow {

public:
    /**
     * @param width    窗口宽度
     * @param height   窗口高度
     * @param titlePtr 窗口名称
     */
    MainWindow(int width, int height, const char* titlePtr);
    virtual ~MainWindow();

public:
    /**
     * 关于
     */
    void about();
    /**
     * 设置
     */
    void setting();

protected:
    void init() override;

};

/**
 * 设置窗口
 */
class SettingWindow : public LFRWindow {

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

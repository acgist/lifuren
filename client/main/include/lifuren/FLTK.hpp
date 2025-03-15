/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * FLTK
 * 
 * https://www.fltk.org/doc-1.3/index.html
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_CLIENT_FLTK_HPP
#define LFR_HEADER_CLIENT_FLTK_HPP

// 去掉冲突
#ifdef  _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

// 窗口大小
#ifndef LFR_WINDOW_DEFAULT
#define LFR_WINDOW_DEFAULT
#define LFR_WINDOW_WIDTH  1280
#define LFR_WINDOW_HEIGHT 720
#define LFR_DIALOG_WIDTH  780
#define LFR_DIALOG_HEIGHT 512
#endif

#include <set>
#include <string>

#include "FL/Fl_Window.H"

class Fl_Choice;

namespace lifuren {

extern void initFltkService(); // 加载FLTK服务
extern void stopFltkService(); // 关闭FLTK服务

/**
 * 抽象窗口
 */
class Window : public Fl_Window {

protected:
    const char* name; // 窗口名称
    Fl_RGB_Image* windowIcon{ nullptr }; // 窗口图标

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    Window(int width, int height, const char* title);
    virtual ~Window();

public:
    virtual void init(); // 初始化窗口

protected:
    void icon  (); // 设置图标
    void center(); // 窗口居中
    virtual void drawElement(); // 绘制组件
    virtual void bindEvent  (); // 绑定事件
    virtual void fillData   (); // 数据填充

};

/**
 * 主窗口
 */
class MainWindow : public Window {

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    MainWindow(int width, int height, const char* title = "李夫人");
    virtual ~MainWindow();

protected:
    virtual void drawElement() override;
    virtual void bindEvent  () override;

};

/**
 * 配置窗口
 */
class ConfigWindow : public Window {

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    ConfigWindow(int width, int height, const char* title = "配置");
    virtual ~ConfigWindow();

protected:
    virtual void drawElement() override;
    virtual void bindEvent  () override;
    virtual void fillData   () override;

};

/**
 * 关于窗口
 */
class AboutWindow : public Window {

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    AboutWindow(int width, int height, const char* title = "关于");
    virtual ~AboutWindow();

protected:
    virtual void drawElement() override;
    virtual void bindEvent  () override;
    virtual void fillData   () override;

};

/**
 * 乐谱窗口
 */
class MusicScoreWindow : public Window {

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    MusicScoreWindow(int width, int height, const char* title = "乐谱");
    virtual ~MusicScoreWindow();

protected:
    virtual void drawElement() override;
    virtual void bindEvent  () override;
    virtual void fillData   () override;
    
};

/**
 * 选择文件
 * 
 * @param title     窗口标题
 * @param filter    文件过滤：*.{cxx,cpp}
 * @param directory 当前目录
 * 
 * @return 选择文件路径
 */
extern std::string fileChooser(const char* title, const char* filter = "*.*", const char* directory = "");

/**
 * 选择文件同时设置到输入框
 * 
 * @param widget    来源组件
 * @param voidPtr   输入框指针
 * @param title     窗口标题
 * @param filter    文件过滤：*.{cxx,cpp}
 * @param directory 当前目录
 */
extern void fileChooser(Fl_Widget* widget, void* voidPtr, const char* title, const char* filter = "*.*", const char* directory = "");

/**
 * 选择目录
 * 
 * @param title     窗口标题
 * @param directory 当前目录
 * 
 * @return 选择目录路径
 */
extern std::string directoryChooser(const char* title, const char* directory = "");

/**
 * 选择目录同时设置到输入框
 * 
 * @param widget    来源组件
 * @param voidPtr   输入框指针
 * @param title     窗口标题
 * @param directory 当前目录
 */
extern void directoryChooser(Fl_Widget* widget, void* voidPtr, const char* title, const char* directory = "");

} // END OF lifuren

#endif // LFR_HEADER_CLIENT_FLTK_HPP

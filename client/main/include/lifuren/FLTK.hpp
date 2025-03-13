/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * FLTK GUI API
 * 
 * https://www.fltk.org/doc-1.3/index.html
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_CLIENT_FLTK_HPP
#define LFR_HEADER_CLIENT_FLTK_HPP

#ifdef  _WIN32
// 去掉min/max冲突
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif // END OF _WIN32

#ifndef LFR_WINDOW_DEFAULT
#define LFR_WINDOW_DEFAULT
#define LFR_WINDOW_WIDTH  1280
#define LFR_WINDOW_HEIGHT 720
#define LFR_DIALOG_WIDTH  720
#define LFR_DIALOG_HEIGHT 480
#endif

#include <set>
#include <string>

#include "FL/Fl_Window.H"

#include "lifuren/Message.hpp"

class Fl_Choice;

namespace lifuren {

extern void initFltkService(); // 加载FLTK服务
extern void stopFltkService(); // 关闭FLTK服务

/**
 * 抽象窗口
 */
class Window : public Fl_Window {

protected:
    Fl_RGB_Image* windowIcon{ nullptr }; // 窗口图标

public:
    Window(
        int width,  // 窗口宽度
        int height, // 窗口高度
        const char* title // 窗口名称
    );
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
    MainWindow(
        int width,  // 窗口宽度
        int height, // 窗口高度
        const char* title = "李夫人" // 窗口名称
    );
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
    ConfigWindow(
        int width,  // 窗口宽度
        int height, // 窗口高度
        const char* title = "配置" // 窗口名称
    );
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
    AboutWindow(
        int width,  // 窗口宽度
        int height, // 窗口宽度
        const char* title = "关于" // 窗口名称
    );
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
    MusicScoreWindow(
        int width,  // 窗口宽度
        int height, // 窗口高度
        const char* title = "乐谱" // 窗口名称
    );
    virtual ~MusicScoreWindow();

protected:
    virtual void drawElement() override;
    virtual void bindEvent  () override;
    virtual void fillData   () override;
    
};

/**
 * 选择文件
 * 
 * @return 选择文件路径
 */
extern std::string fileChooser(
    const char* title, // 窗口标题
    const char* filter    = "*.*", // 文件过滤：*.{cxx,cpp}
    const char* directory = ""     // 当前目录
);

/**
 * 选择文件同时设置到输入框
 */
extern void fileChooser(
    Fl_Widget * widget,  // 来源组件
    void      * voidPtr, // 输入框指针
    const char* title,   // 窗口标题
    const char* filter    = "*.*", // 文件过滤：*.{cxx,cpp}
    const char* directory = ""     // 当前目录
);

/**
 * 选择目录
 * 
 * @return 选择目录路径
 */
extern std::string directoryChooser(
    const char* title, // 窗口标题
    const char* directory = "" // 当前目录
);

/**
 * 选择目录同时设置到输入框
 */
extern void directoryChooser(
    Fl_Widget * widget,  // 来源组件
    void      * voidPtr, // 输入框指针
    const char* title,   // 窗口标题
    const char* directory = "" // 当前目录
);

/**
 * 填充选择框
 */
extern void fillChoice(
    Fl_Choice* choice, // 选择框
    const std::set<std::string>& set, // 选项列表
    const std::string& value = ""     // 默认选项
);

} // END OF lifuren

#endif // LFR_HEADER_CLIENT_FLTK_HPP

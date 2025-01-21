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
#ifndef LFR_HEADER_BOOT_FLTK_HPP
#define LFR_HEADER_BOOT_FLTK_HPP

#ifdef  _WIN32
// 去掉min/max冲突
#ifndef NOMINMAX
#define NOMINMAX
#endif
// https://github.com/yhirose/cpp-httplib
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif

#ifndef LFR_WINDOW_DEFAULT
#define LFR_WINDOW_DEFAULT
#define LFR_WINDOW_WIDTH  1280
#define LFR_WINDOW_HEIGHT 820
#endif

#include <set>
#include <string>

#include "FL/Fl_Window.H"

#include "lifuren/Message.hpp"

class Fl_Choice;

namespace lifuren {

extern void initFltkService();     // 加载FLTK服务
extern void shutdownFltkService(); // 关闭FLTK服务

/**
 * 抽象窗口
 */
class Window : public Fl_Window {

protected:
    Fl_RGB_Image* windowIcon{ nullptr }; // 窗口图标

public:
    Window(
        int width,        // 窗口宽度
        int height,       // 窗口高度
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
 * 配置管理
 */
class Configuration {

public:
    virtual ~Configuration();

protected:
    virtual void saveConfig(); // 保存配置

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
 * 音频生成窗口
 */
class AudioWindow : public Window, public Configuration {

public:
    AudioWindow(
        int width,  // 窗口宽度
        int height, // 窗口高度
        const char* title = "音频生成" // 窗口名称
    );
    virtual ~AudioWindow();

protected:
    virtual void drawElement() override;
    virtual void bindEvent  () override;
    virtual void fillData   () override;

};

/**
 * 视频生成窗口
 */
class VideoWindow : public Window, public Configuration {

public:
    VideoWindow(
        int width,  // 窗口宽度
        int height, // 窗口高度
        const char* title = "视频生成" // 窗口名称
    );
    virtual ~VideoWindow();

protected:
    virtual void drawElement() override;
    virtual void bindEvent  () override;
    virtual void fillData   () override;

};

/**
 * 诗词生成窗口
 */
class PoetryWindow : public Window, public Configuration {

public:
    PoetryWindow(
        int width,  // 窗口宽度
        int height, // 窗口高度
        const char* title = "诗词生成" // 窗口名称
    );
    virtual ~PoetryWindow();

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
 * 后台任务窗口
 */
class ThreadWindow : public Window {

public:
    // 是否可以关闭：任务执行完成后隐藏时释放资源
    bool closeable { false };
    // 后台任务类型
    lifuren::message::Type type { lifuren::message::Type::NONE };

public:
    ThreadWindow(
        int width,  // 窗口宽度
        int height, // 窗口高度
        const char* title = "后台任务" // 窗口名称
    );
    virtual ~ThreadWindow();

protected:
    virtual void drawElement() override;
    virtual void bindEvent  () override;

public:
    static void showThread (lifuren::message::Type type); // 显示任务
    static bool checkThread(lifuren::message::Type type); // 判断是否含有相同类型任务
    static bool startThread(lifuren::message::Type type, const char* title, std::function<void()> task, std::function<void()> callback = nullptr); // 开始任务
    static bool stopThread (lifuren::message::Type type); // 结束任务
    static bool checkAudioThread (); // 判断是否含有音频任务
    static bool checkVideoThread (); // 判断是否含有视频任务
    static bool checkPoetryThread(); // 判断是否含有诗词任务

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

#endif // LFR_HEADER_BOOT_FLTK_HPP

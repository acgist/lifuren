/**
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

// https://github.com/yhirose/cpp-httplib
#ifdef  _WIN32
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
class Fl_RGB_Image;

namespace lifuren {

/**
 * 加载FLTK窗口
 */
extern void initFltkWindow();

/**
 * 关闭FLTK窗口
 */
extern void shutdownFltkWindow();

/**
 * 抽象窗口
 */
class Window : public Fl_Window {

protected:
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
    /**
     * 初始化窗口
     */
    virtual void init();

protected:
    /**
     * 绘制组件
     * 绑定事件
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

/**
 * 配置管理
 */
class Configuration {

public:
    /**
     * 保存配置
     */
    virtual void saveConfig();
    /**
     * 重新绘制配置组件
     */
    virtual void redrawConfigElement() = 0;

};

class MainWindow;
class AudioWindow;
class ImageWindow;
class VideoWindow;
class PoetryWindow;
class AboutWindow;

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

};

/**
 * 音频生成窗口
 */
class AudioWindow : public Window, public Configuration {

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    AudioWindow(int width, int height, const char* title = "音频生成");
    virtual ~AudioWindow();

public:
    virtual void saveConfig() override;
    virtual void redrawConfigElement() override;
    
protected:
    virtual void drawElement() override;

};

/**
 * 图片生成窗口
 */
class ImageWindow : public Window, public Configuration {

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    ImageWindow(int width, int height, const char* title = "图片生成");
    virtual ~ImageWindow();

public:
    virtual void saveConfig() override;
    virtual void redrawConfigElement() override;
    
protected:
    virtual void drawElement() override;

};

/**
 * 视频生成窗口
 */
class VideoWindow : public Window, public Configuration {

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    VideoWindow(int width, int height, const char* title = "视频生成");
    virtual ~VideoWindow();

public:
    virtual void saveConfig() override;
    virtual void redrawConfigElement() override;
    
protected:
    virtual void drawElement() override;

};

/**
 * 诗词生成窗口
 */
class PoetryWindow : public Window, public Configuration {

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    PoetryWindow(int width, int height, const char* title = "诗词生成");
    virtual ~PoetryWindow();

public:
    virtual void saveConfig() override;
    virtual void redrawConfigElement() override;

protected:
    virtual void drawElement() override;

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

};

/**
 * 后台任务窗口
 */
class ThreadWindow : public Window {

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    ThreadWindow(int width, int height, const char* title = "后台任务");
    virtual ~ThreadWindow();

protected:
    virtual void drawElement() override;

public:
    static bool hasThread  (lifuren::message::Type type);
    static void showThread (lifuren::message::Type type);
    static bool checkThread(lifuren::message::Type type);
    static bool startThread(lifuren::message::Type type, const char* title, std::function<void()> task, bool notify = false);
    static bool stopThread (lifuren::message::Type type);

};

/**
 * @param title     窗口标题
 * @param filter    文件过滤：*.{cxx,cpp}
 * @param directory 当前目录
 * 
 * @return 选择文件路径
 */
extern std::string fileChooser(const char* title, const char* filter = "*.*", const char* directory = "");

/**
 * @param title     窗口标题
 * @param directory 当前目录
 * 
 * @return 选择目录路径
 */
extern std::string directoryChooser(const char* title, const char* directory = "");

/**
 * @param choice 选择框
 * @param set    选项列表
 * @param value  默认选项
 */
extern void fillChoice(Fl_Choice* choice, const std::set<std::string>& set, const std::string& value = "");

} // END lifuren

#endif // LFR_HEADER_BOOT_FLTK_HPP

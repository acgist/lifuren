/**
 * FLTK GUI
 * 
 * 文档：https://www.fltk.org/doc-1.3/index.html
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_BOOT_FLTK_HPP
#define LFR_HEADER_BOOT_FLTK_HPP

#ifdef  _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif

#include <set>
#include <string>

#include "FL/Fl_Window.H"

#ifndef LFR_WINDOW_DEFAULT
#define LFR_WINDOW_DEFAULT
#define LFR_WINDOW_WIDTH  1280
#define LFR_WINDOW_HEIGHT 820
#endif

#ifndef LFR_CHOICE_PATH
#if _WIN32
#define LFR_CHOICE_PATH(path) \
lifuren::strings::replace(path, "\\", "\\\\");
#else
#define LFR_CHOICE_PATH(path) \
lifuren::strings::replace(path, "/", "\\/");
#endif
#endif

class Fl_Choice;
class Fl_PNG_Image;

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
    // 图标指针
    Fl_PNG_Image* iconImagePtr{ nullptr };

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    Window(int width, int height, const char* title);
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

};

class MainWindow;
class MarkWindow;
class ImageWindow;
class PoetryWindow;
class AboutWindow;

/**
 * 配置管理
 */
class Configuration {

protected:
    // 保存配置
    virtual void saveConfig();
    // 重新绘制配置元素
    virtual void redrawConfigElement() = 0;

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

};

/**
 * 诗词标记窗口
 */
class MarkWindow : public Window, public Configuration {

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    MarkWindow(int width, int height, const char* title = "诗词标记");
    virtual ~MarkWindow();

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
    ImageWindow(int width, int height, const char* title = "图片内容生成");
    virtual ~ImageWindow();

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
    PoetryWindow(int width, int height, const char* title = "诗词内容生成");
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
 * @param title     标题
 * @param directory 当前目录
 * @param filter    文件过滤：*.{cxx,cpp}
 * 
 * @return 选择文件路径
 */
extern std::string fileChooser(const char* title, const char* directory = "", const char* filter = "*.*");

/**
 * @param title     标题
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

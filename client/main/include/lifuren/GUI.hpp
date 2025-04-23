/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * GUI
 * 
 * https://www.wxwidgets.org/
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_CLIENT_GUI_HPP
#define LFR_HEADER_CLIENT_GUI_HPP

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
#define LFR_DIALOG_HEIGHT 520
#endif

#include <set>
#include <string>

#include "wx/app.h"
#include "wx/frame.h"

namespace lifuren {

extern void initGUI(); // 加载GUI

/**
 * 应用
 */
class Application : public wxApp {

public:
    bool OnInit() override;

};

/**
 * 抽象窗口
 */
class Window : public wxFrame {

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    Window(int width, int height, const wxString& title);
    virtual ~Window();

public:
    virtual void init(); // 初始化窗口

protected:
    virtual void loadIcon  (); // 设置图标
    virtual void drawWidget(); // 绘制组件
    virtual void bindEvent (); // 绑定事件
    virtual void fillData  (); // 数据填充

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
    MainWindow(int width, int height, const wxString& title = L"李夫人");
    ~MainWindow();

protected:
    void drawWidget() override;
    void bindEvent () override;

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
    ConfigWindow(int width, int height, const wxString& title = L"配置");
    virtual ~ConfigWindow();

protected:
    virtual void drawWidget() override;
    virtual void bindEvent () override;
    virtual void fillData  () override;

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
    AboutWindow(int width, int height, const wxString& title = L"关于");
    virtual ~AboutWindow();

protected:
    virtual void drawWidget() override;
    virtual void bindEvent () override;
    virtual void fillData  () override;

};

/**
 * @param title     窗口标题
 * @param filter    文件过滤：*.{cxx,cpp}
 * @param directory 当前目录
 * 
 * @return 选择文件路径
 */
extern std::string file_chooser(const wxString& title, const wxString& filter = "", const wxString& directory = "");

/**
 * @param title     窗口标题
 * @param directory 当前目录
 * 
 * @return 选择目录路径
 */
extern std::string directory_chooser(const wxString& title, const wxString& directory = "");

/**
 * @param path 相对路径
 * 
 * @return 绝对路径
 */
extern wxString app_base_dir(const wxString& path);

} // END OF lifuren

#endif // LFR_HEADER_CLIENT_GUI_HPP

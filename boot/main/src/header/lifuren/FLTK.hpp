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

#include <string>

#include "FL/Fl.H"
#include "FL/Fl_Window.H"
#include "FL/Fl_PNG_Image.H"

#include "lifuren/Ptr.hpp"
#include "lifuren/Client.hpp"

#ifndef LFR_WINDOW_DEFAULT
#define LFR_WINDOW_DEFAULT
#define LFR_WINDOW_WIDTH         1280
#define LFR_WINDOW_WIDTH_CONFIG  820
#define LFR_WINDOW_HEIGHT        800
#define LFR_WINDOW_HEIGHT_CONFIG 820
#endif

#ifndef LFR_CHOICE_SET_DEFAULT
#define LFR_CHOICE_SET_DEFAULT(widget, defaultValue) \
widget->value(widget->find_index(defaultValue.c_str()));
#endif

#ifndef LFR_CHOICE_GET_DEFAULT
#define LFR_CHOICE_GET_DEFAULT(target, source) \
target = source->value() >= 0 ? source->text() : ""
#endif

// 下拉选择
#ifndef LFR_CHOICE_ADD_LIST
#define LFR_CHOICE_ADD_LIST(x, y, map, choicePtr, groupName, labelName, defaultValue)         \
{                                                                                             \
    choicePtr = new Fl_Choice(x, y, 80, 30, labelName);                                       \
    choicePtr->add(defaultValue);                                                             \
    auto iterator = lifuren::map.find(groupName);                                             \
    if(iterator != lifuren::map.end()) {                                                      \
        std::vector<LabelFile>& vector = iterator->second;                                    \
        std::for_each(vector.begin(), vector.end(), [this](auto& label) {                     \
            if(label.name == labelName) {                                                     \
                std::for_each(label.labels.begin(), label.labels.end(), [this](auto& value) { \
                    choicePtr->add(value.c_str());                                            \
                });                                                                           \
            }                                                                                 \
        });                                                                                   \
    }                                                                                         \
    auto defaultIndex = choicePtr->find_index(defaultValue);                                  \
    choicePtr->value(defaultIndex);                                                           \
}
#endif

#ifndef LFR_CHOICE_TRANSFER
#if _WIN32
#define LFR_CHOICE_TRANSFER(path) \
lifuren::strings::replace(path, "\\", "\\\\");
#else
#define LFR_CHOICE_TRANSFER(path) \
lifuren::strings::replace(path, "/", "\\/");
#endif
#endif

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
    Fl_PNG_Image* iconImagePtr = nullptr;

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    Window(int width, int height, const char* title);
    // 析构函数
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
class ChatWindow;
class ChatConfigWindow;
class ImageWindow;
class ImageConfigWindow;
class PoetryWindow;
class PoetryConfigWindow;
class AboutWindow;
class ImageMarkWindow;
class PoetryMarkWindow;
class DocumentMarkWindow;
class FinetuneWindow;
class QuantizationWindow;

/**
 * 配置管理
 */
class Configuration {

protected:
    /**
     * 保存配置
     */
    virtual void saveConfig();
    /**
     * 重新绘制配置元素
     */
    virtual void redrawConfigElement() = 0;

};

/**
 * 配置窗口
 */
class ConfigWindow : public Window, public Configuration {

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    ConfigWindow(int width, int height, const char* title = "配置");
    // 析构函数
    virtual ~ConfigWindow();

};

/**
 * 主窗口
 */
class MainWindow : public Window {

private:
    // 图片标记窗口
    ImageMarkWindow*    imageMarkWindowPtr    = nullptr;
    // 诗词标记窗口
    PoetryMarkWindow*   poetryMarkWindowPtr   = nullptr;
    // 文档标记窗口
    DocumentMarkWindow* documentMarkWindowPtr = nullptr;
    // 模型微调窗口
    FinetuneWindow*     finetuneWindowPtr     = nullptr;
    // 模型量化窗口
    QuantizationWindow* quantizationWindowPtr = nullptr;
    // 聊天窗口
    ChatWindow*  chatWindowPtr  = nullptr;
    // 图片生成窗口
    ImageWindow* imageWindowPtr = nullptr;
    // 诗词生成窗口
    PoetryWindow* poetryWindowPtr = nullptr;
    // 关于窗口
    AboutWindow* aboutWindowPtr = nullptr;

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    MainWindow(int width, int height, const char* title);
    // 析构函数
    virtual ~MainWindow();

public:
    // 图片标记
    void imageMark();
    // 诗词标记
    void poetryMark();
    // 文档标记
    void documentMark();
    // 模型微调
    void finetune();
    // 模型量化
    void quantization();
    // 对话
    void chat();
    // 图片内容生成
    void image();
    // 诗词内容生成
    void poetry();
    // 关于
    void about();

protected:
    virtual void drawElement() override;

};

/**
 * 标记窗口
 */
class MarkWindow : public Window {

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    MarkWindow(int width, int height, const char* title = "标记");
    // 析构函数
    virtual ~MarkWindow();

protected:
    /**
     * 下一个标记
     */
    virtual void prevMark();
    /**
     * 上一个标记
     */
    virtual void nextMark();

};

/**
 * 图片标记窗口
 */
class ImageMarkWindow : public MarkWindow, public Configuration {

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    ImageMarkWindow(int width, int height, const char* title = "图片标记");
    // 析构函数
    virtual ~ImageMarkWindow();

public:
    virtual void saveConfig() override;
    virtual void redrawConfigElement() override;

protected:
    virtual void drawElement() override;

};

/**
 * 诗词标记窗口
 */
class PoetryMarkWindow : public MarkWindow, public Configuration {

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    PoetryMarkWindow(int width, int height, const char* title = "诗词标记");
    // 析构函数
    virtual ~PoetryMarkWindow();

public:
    virtual void saveConfig() override;
    virtual void redrawConfigElement() override;
    
protected:
    virtual void drawElement() override;

};

/**
 * 文档标记窗口
 */
class DocumentMarkWindow : public MarkWindow, public Configuration {

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    DocumentMarkWindow(int width, int height, const char* title = "文档标记");
    // 析构函数
    virtual ~DocumentMarkWindow();

public:
    virtual void saveConfig() override;
    virtual void redrawConfigElement() override;

protected:
    virtual void drawElement() override;

};

/**
 * 模型微调窗口
 */
class FinetuneWindow : public Window {

private:

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    FinetuneWindow(int width, int height, const char* title = "模型微调");
    // 析构函数
    virtual ~FinetuneWindow();

protected:
    virtual void drawElement() override;

};

/**
 * 模型量化窗口
 */
class QuantizationWindow : public Window {

private:

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    QuantizationWindow(int width, int height, const char* title = "模型量化");
    // 析构函数
    virtual ~QuantizationWindow();

protected:
    virtual void drawElement() override;

};

/**
 * 模型窗口
 */
class ModelWindow : public Window {

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    ModelWindow(int width, int height, const char* title);
    // 析构函数
    virtual ~ModelWindow();

};

/**
 * 聊天
 */
class ChatWindow : public ModelWindow {

public:
    // 聊天终端
    ChatClient* clientPtr{ nullptr };

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    ChatWindow(int width, int height, const char* title = "聊天");
    // 析构函数
    virtual ~ChatWindow();

protected:
    virtual void drawElement() override;

};

/**
 * 聊天配置
 */
class ChatConfigWindow : public ConfigWindow {

public:
    ChatConfigWindow(int width, int height, const char* title = "聊天配置");
    virtual ~ChatConfigWindow();

public:
    virtual void redrawConfigElement() override;

protected:
    virtual void drawElement() override;
    virtual void saveConfig()  override;

};

/**
 * 图片内容生成
 */
class ImageWindow : public ModelWindow {

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    ImageWindow(int width, int height, const char* title = "图片内容生成");
    // 析构函数
    virtual ~ImageWindow();

protected:
    virtual void drawElement() override;

};

/**
 * 诗词内容生成
 */
class PoetryWindow : public ModelWindow {

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    PoetryWindow(int width, int height, const char* title = "诗词内容生成");
    // 析构函数
    virtual ~PoetryWindow();

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
    // 析构函数
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
extern std::string fileChooser(const char* title, const char* directory = ".", const char* filter = "*.*");

/**
 * @param title     标题
 * @param directory 当前目录
 * 
 * @return 选择目录路径
 */
extern std::string directoryChooser(const char* title, const char* directory = ".");

} // END lifuren

#endif // LFR_HEADER_BOOT_FLTK_HPP

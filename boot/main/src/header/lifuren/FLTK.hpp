/**
 * FLTK GUI
 * 
 * REST.hpp必须在FLTK.hpp的前面
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_BOOT_FLTK_HPP
#define LFR_HEADER_BOOT_FLTK_HPP

#include <string>

#include "FL/Fl.H"
#include "FL/Fl_Input.H"
#include "FL/Fl_Button.H"
#include "FL/Fl_Choice.H"
#include "FL/Fl_Window.H"
#include "FL/Fl_PNG_Image.H"
#include "FL/Fl_JPEG_Image.H"
#include "FL/Fl_Text_Buffer.H"
#include "FL/Fl_Text_Display.H"

#include "lifuren/Ptr.hpp"
#include "lifuren/config/Config.hpp"

// 窗口宽度
#ifndef LFR_HALF_WIDTH
#define LFR_HALF_WIDTH(padding) (this->w() - padding) / 2
#endif

// 图片缩放
#ifndef LFR_IMAGE_PREVIEW_SCALE
#define LFR_IMAGE_PREVIEW_SCALE 1.2
#endif

// 目录选择
#ifndef LFR_INPUT_DIRECTORY_CHOOSER
#define LFR_INPUT_DIRECTORY_CHOOSER(inputPtr, configPtr, configName, windowName)   \
    this->inputPtr->callback([](Fl_Widget* widgetPtr, void* voidPtr) {             \
        const char* value = ((windowName*) voidPtr)->inputPtr->value();            \
        if(value == nullptr || std::strlen(value) == 0) {                          \
            return;                                                                \
        }                                                                          \
        ((windowName*) voidPtr)->configPtr->configName = value;                    \
    }, this);
#endif

// 目录选择并且回调
#ifndef LFR_INPUT_DIRECTORY_CHOOSER_CALLBACK
#define LFR_INPUT_DIRECTORY_CHOOSER_CALLBACK(inputPtr, configPtr, configName, windowName, callbackName) \
    this->inputPtr->callback([](Fl_Widget* widgetPtr, void* voidPtr) {                                  \
        const char* value = ((windowName*) voidPtr)->inputPtr->value();                                 \
        if(value == nullptr || std::strlen(value) == 0) {                                               \
            return;                                                                                     \
        }                                                                                               \
        ((windowName*) voidPtr)->configPtr->configName = value;                                         \
        callbackName(value);                                                                            \
    }, this);
#endif

// 下拉选择
#ifndef LFR_CHOICE_BUTTON
#define LFR_CHOICE_BUTTON(x, y, map, buttonPtr, groupName, labelName, defaultValue)           \
{                                                                                             \
    buttonPtr = new Fl_Choice(x, y, 80, 30, labelName);                                       \
    buttonPtr->add(defaultValue);                                                             \
    auto iterator = lifuren::map.find(groupName);                                             \
    if(iterator != lifuren::map.end()) {                                                      \
        std::vector<LabelFile>& vector = iterator->second;                                    \
        std::for_each(vector.begin(), vector.end(), [this](auto& label) {                     \
            if(label.name == labelName) {                                                     \
                std::for_each(label.labels.begin(), label.labels.end(), [this](auto& value) { \
                    buttonPtr->add(value.c_str());                                            \
                });                                                                           \
            }                                                                                 \
        });                                                                                   \
    }                                                                                         \
    auto defaultPtr = buttonPtr->find_item(defaultValue);                                     \
    buttonPtr->value(defaultPtr);                                                             \
}
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

// 功能窗口
class MainWindow;
class ChatWindow;
class ImageWindow;
class VideoWindow;
class AboutWindow;
class DocsMarkWindow;
class ImageMarkWindow;
class PoetryMarkWindow;
class FinetuneWindow;
class QuantizationWindow;

/**
 * 主窗口
 */
class MainWindow : public Window {

private:
    // 文档标记按钮
    Fl_Button* docsMarkButtonPtr = nullptr;
    // 文档标记窗口
    DocsMarkWindow* docsMarkWindowPtr = nullptr;
    // 图片标记按钮
    Fl_Button* imageMarkButtonPtr = nullptr;
    // 图片标记窗口
    ImageMarkWindow* imageMarkWindowPtr = nullptr;
    // 诗词标记按钮
    Fl_Button* poetryMarkButtonPtr = nullptr;
    // 诗词标记窗口
    PoetryMarkWindow* poetryMarkWindowPtr = nullptr;
    // 模型微调按钮
    Fl_Button* finetuneButtonPtr = nullptr;
    // 模型微调窗口
    FinetuneWindow* finetuneWindowPtr = nullptr;
    // 模型量化按钮
    Fl_Button* quantizationButtonPtr = nullptr;
    // 模型量化窗口
    QuantizationWindow* quantizationWindowPtr = nullptr;
    // 聊天按钮
    Fl_Button* chatButtonPtr = nullptr;
    // 聊天窗口
    ChatWindow* chatWindowPtr = nullptr;
    // 图片生成按钮
    Fl_Button* imageButtonPtr = nullptr;
    // 图片生成窗口
    ImageWindow* imageWindowPtr = nullptr;
    // 视频生成按钮
    Fl_Button* videoButtonPtr = nullptr;
    // 视频生成窗口
    VideoWindow* videoWindowPtr = nullptr;
    // 重新加载配置按钮
    Fl_Button* reloadButtonPtr = nullptr;
    // 关于按钮
    Fl_Button* aboutButtonPtr = nullptr;
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
    // 文档标记
    void docsMark();
    // 图片标记
    void imageMark();
    // 诗词标记
    void poetryMark();
    // 模型微调
    void finetune();
    // 模型量化
    void quantization();
    // 对话
    void chat();
    // 图片内容生成
    void image();
    // 视频内容生成
    void video();
    // 关于
    void about();

protected:
    // 加载组件
    virtual void drawElement() override;

};

/**
 * 标记窗口
 */
class MarkWindow : public Window {

public:
    // 上个内容
    Fl_Button* prevPtr = nullptr;
    // 下个内容
    Fl_Button* nextPtr = nullptr;
    // 数据路径
    Fl_Input* datasetPathPtr = nullptr;

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    MarkWindow(int width, int height, const char* title = "关于");
    // 析构函数
    virtual ~MarkWindow();

};

/**
 * 文档标记窗口
 */
class DocsMarkWindow : public MarkWindow {

public:
    // 配置
    lifuren::config::DocsMark* docsConfigPtr{ nullptr };

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    DocsMarkWindow(int width, int height, const char* title = "文档标记");
    // 析构函数
    virtual ~DocsMarkWindow();

protected:
    // 加载组件
    virtual void drawElement() override;

};

/**
 * 图片标记窗口
 */
class ImageMarkWindow : public MarkWindow {

public:
    // 配置
    lifuren::config::ImageMark* imageConfigPtr{ nullptr };

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    ImageMarkWindow(int width, int height, const char* title = "图片标记");
    // 析构函数
    virtual ~ImageMarkWindow();

protected:
    // 加载组件
    virtual void drawElement() override;

};

/**
 * 诗词标记窗口
 */
class PoetryMarkWindow : public MarkWindow {

public:
    // 自动标记：通过已有标记自动标记
    Fl_Button* autoMarkPtr = nullptr;
    // 配置
    lifuren::config::PoetryMark* poetryConfigPtr{ nullptr };

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    PoetryMarkWindow(int width, int height, const char* title = "诗词标记");
    // 析构函数
    virtual ~PoetryMarkWindow();

protected:
    // 加载组件
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
    // 加载组件
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
    // 加载组件
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

protected:
    // 模型名称
    Fl_Choice* modelPtr = nullptr;
    // 开始训练
    Fl_Button* trainStartPtr = nullptr;
    // 结束训练
    Fl_Button* trainStopPtr = nullptr;

};

/**
 * 聊天
 */
class ChatWindow : public ModelWindow {

public:
    // 配置
    lifuren::config::Chat* chatConfigPtr{ nullptr };

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
    // 加载组件
    virtual void drawElement() override;

};

/**
 * 图片内容生成
 */
class ImageWindow : public ModelWindow {

public:
    // 配置
    lifuren::config::Image* imageConfigPtr{ nullptr };

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
    // 加载组件
    virtual void drawElement() override;

};

/**
 * 视频内容生成
 */
class VideoWindow : public ModelWindow {

public:
    // 配置
    lifuren::config::Video* videoConfigPtr{ nullptr };

public:
    /**
     * @param width  窗口宽度
     * @param height 窗口高度
     * @param title  窗口名称
     */
    VideoWindow(int width, int height, const char* title = "视频内容生成");
    // 析构函数
    virtual ~VideoWindow();

protected:
    // 加载组件
    virtual void drawElement() override;

};

/**
 * 关于窗口
 */
class AboutWindow : public Window {

private:
    // 官网
    Fl_Button* homePagePtr = nullptr;
    // 关于内容
    Fl_Text_Buffer* aboutBufferPtr = nullptr;
    // 关于组件
    Fl_Text_Display* aboutDisplayPtr = nullptr;

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
    // 加载组件
    virtual void drawElement() override;

};

/**
 * 目录选择组件
 */
class Fl_Input_Directory_Chooser : public Fl_Input {

private:
    // 标题
    const char* title = nullptr;
    // 目录
    const char* directory = nullptr;

public:
    /**
     * @param x         x
     * @param y         y
     * @param width     宽度
     * @param height    高度
     * @param title     标题
     * @param directory 当前目录
     */
    Fl_Input_Directory_Chooser(
        int x,
        int y,
        int width,
        int height,
        const char* title,
        const char* directory = "."
    );
    // 析构函数
    virtual ~Fl_Input_Directory_Chooser();

public:
    /**
     * @param event 事件
     */
    int handle(int event) override;

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

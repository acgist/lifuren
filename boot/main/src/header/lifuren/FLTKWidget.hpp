/**
 * FLTK组件
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_BOOT_FLTK_WIDGET_HPP
#define LFR_HEADER_BOOT_FLTK_WIDGET_HPP

#include "FL/Fl_Input.H"

namespace lifuren {

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

} // END OF lifuren

#endif // END OF LFR_HEADER_BOOT_FLTK_WIDGET_HPP
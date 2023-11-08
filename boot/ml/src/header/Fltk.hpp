#pragma once

#include <iostream>

#include "GLog.hpp"

#include "FL/Fl.H"
#include "FL/Fl_Input.H"
#include "FL/Fl_Button.H"
#include "FL/Fl_Window.H"

namespace lifuren {

class LifurenWindow : public Fl_Window {

public:
    /**
     * @param width    窗口宽度
     * @param height   窗口高度
     * @param titlePtr 窗口名称
     */
    LifurenWindow(int width, int height, const char* titlePtr);
    /**
     * 加载控件
     */
    void init();
    /**
     * 静态函数
     */
    static void buttonCallback(Fl_Widget* widgetPtr, void* voidPtr);
};

}
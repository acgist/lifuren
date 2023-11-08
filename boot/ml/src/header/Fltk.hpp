/**
 * 窗口
 * 
 * @author acgist
 */
#pragma once

#include <iostream>

#include "GLog.hpp"

#include "FL/Fl.H"
#include "FL/Fl_Input.H"
#include "FL/Fl_Button.H"
#include "FL/Fl_Window.H"

namespace lifuren {

/**
 * 窗口
 */
class LifurenWindow : public Fl_Window {

public:
    /**
     * 输入框
     */
    Fl_Input* inputPtr;
    /**
     * 按钮
     */
    Fl_Button* buttonPtr;
    /**
     * 按钮
     */
    Fl_Button* buttonProxyPtr;

public:
    ~LifurenWindow() {
        delete this->inputPtr;
        delete this->buttonPtr;
        delete this->buttonProxyPtr;
    };
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
     * 按钮回调函数
     * 如果想要按钮直接调用需要改为静态函数
     * 
     * @param widgetPtr 窗口指针
     * @param voidPtr   当前窗口指针
     */
    void buttonCallback(Fl_Widget* widgetPtr, void* voidPtr);
};

}
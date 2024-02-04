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
 * 
 * @author acgist
 */
class LifurenWindow : public Fl_Window {

public:
    /**
     * 输入框
     */
    Fl_Input*  inputPtr;
    /**
     * 按钮
     */
    Fl_Button* buttonPtr;
    /**
     * 代理按钮
     */
    Fl_Button* buttonProxyPtr;

public:
    /**
     * 构造函数
     * 
     * @param width    窗口宽度
     * @param height   窗口高度
     * @param titlePtr 窗口名称
     */
    LifurenWindow(int width, int height, const char* titlePtr);
    /**
     * 析构函数
     */
    ~LifurenWindow();

public:
    /**
     * 初始化窗口
     */
    void init();
    /**
     * 按钮回调
     * 
     * @param widgetPtr 窗口
     * @param voidPtr   事件源
     */
    void buttonCallback(Fl_Widget* widgetPtr, void* voidPtr);
    
};

}
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
    Fl_Input*  inputPtr;
    Fl_Button* buttonPtr;
    Fl_Button* buttonProxyPtr;

public:
    ~LifurenWindow();
    LifurenWindow(int width, int height, const char* titlePtr);

public:
    void init();
    void buttonCallback(Fl_Widget* widgetPtr, void* voidPtr);
    
};

}
#include "lifuren/FLTK.hpp"

#include "FL/Fl_Input.H"
#include "FL/Fl_Button.H"

#include "spdlog/spdlog.h"

static Fl_Button* prevPtr{ nullptr };
static Fl_Button* nextPtr{ nullptr };
static Fl_Input*  datasetPathPtr{ nullptr };

lifuren::DocsMarkWindow::DocsMarkWindow(int width, int height, const char* title) : MarkWindow(width, height, title) {
}

lifuren::DocsMarkWindow::~DocsMarkWindow() {
    SPDLOG_DEBUG("关闭窗口：{}", __FILE__);
    LFR_DELETE_PTR(prevPtr);
    LFR_DELETE_PTR(nextPtr);
    LFR_DELETE_PTR(datasetPathPtr);
}

void lifuren::DocsMarkWindow::drawElement() {
}

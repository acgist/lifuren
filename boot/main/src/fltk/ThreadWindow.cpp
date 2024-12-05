#include "lifuren/FLTK.hpp"

#include <map>
#include <memory>
#include <thread>
#include <vector>

#include "lifuren/Raii.hpp"
#include "lifuren/Thread.hpp"
#include "lifuren/Message.hpp"

#include "FL/Fl_Button.H"
#include "FL/Fl_Text_Buffer.H"
#include "FL/Fl_Text_Display.H"

static Fl_Button      * cancel { nullptr };
static Fl_Text_Buffer * buffer { nullptr };
static Fl_Text_Display* display{ nullptr };

// 映射
static std::map<lifuren::message::Type, std::shared_ptr<lifuren::thread::ThreadWorker>> thread_worker;
// 依赖树
static std::map<lifuren::message::Type, std::vector<lifuren::message::Type>> depend_tree;

lifuren::ThreadWindow::ThreadWindow(int width, int height, const char* title) : Window(width, height, title) {
}

lifuren::ThreadWindow::~ThreadWindow() {
    LFR_DELETE_PTR(cancel);
    LFR_DELETE_PTR(buffer);
    LFR_DELETE_PTR(display);
    // 取消回调
}

void lifuren::ThreadWindow::drawElement() {
    
}

bool lifuren::ThreadWindow::hasThread(lifuren::message::Type type) {
    return thread_worker.contains(type);
}

void lifuren::ThreadWindow::showThread(lifuren::message::Type type) {
}

bool lifuren::ThreadWindow::checkThread(lifuren::message::Type type) {
    return true;
}

bool lifuren::ThreadWindow::startThread(lifuren::message::Type type, bool notify) {
    return true;
}

bool lifuren::ThreadWindow::stopThread(lifuren::message::Type type) {
    return true;
}

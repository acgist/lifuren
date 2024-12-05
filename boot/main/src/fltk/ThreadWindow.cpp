#include "lifuren/FLTK.hpp"

#include <map>
#include <memory>
#include <thread>
#include <vector>

#include "spdlog/spdlog.h"

#include "lifuren/Raii.hpp"
#include "lifuren/Thread.hpp"
#include "lifuren/Message.hpp"

#include "FL/fl_ask.H"
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
    auto iter = thread_worker.find(type);
    if(iter == thread_worker.end()) {
        return;
    }
    if(iter->second->source) {
        static_cast<ThreadWindow*>(iter->second->source)->show();
    }
}

bool lifuren::ThreadWindow::checkThread(lifuren::message::Type type) {
    if(hasThread(type)) {
        // TODO：提示？
        showThread(type);
        return false;
    }
    return true;
}

bool lifuren::ThreadWindow::startThread(lifuren::message::Type type, const char* title, std::function<void()> task, bool notify) {
    if(!checkThread(type)) {
        return false;
    }
    ThreadWindow* window = new ThreadWindow(800, 600, title);
    window->callback([](Fl_Widget*, void* voidPtr) {
        ThreadWindow* thisWindow = static_cast<ThreadWindow*>(voidPtr);
        thisWindow->hide();
        delete thisWindow;
    }, window);
    auto worker = std::make_shared<lifuren::thread::ThreadWorker>();
    thread_worker.emplace(type, worker);
    worker->stop   = false;
    worker->source = window;
    worker->thread = std::make_shared<std::thread>([type, task, worker, title, notify]() {
        lifuren::thread::ThreadWorker::fltk_thread = true;
        lifuren::thread::ThreadWorker::this_thread_worker = worker.get();
        try {
            task();
        } catch(const std::exception& e) {
            SPDLOG_ERROR("任务执行异常：{} - {}", title, e.what());
        }
        lifuren::thread::ThreadWorker::this_thread_worker = nullptr;
        if(notify) {
            fl_message("任务完成：{}", title);
        }
        thread_worker.erase(type);
    });
    worker->thread->detach();
    return true;
}

bool lifuren::ThreadWindow::stopThread(lifuren::message::Type type) {
    auto iter = thread_worker.find(type);
    if(iter == thread_worker.end()) {
        return false;
    }
    iter->second->stop = true;
    return true;
}

#include "lifuren/FLTK.hpp"

#include <map>
#include <memory>
#include <thread>
#include <vector>

#include "spdlog/spdlog.h"

#include "lifuren/Raii.hpp"
#include "lifuren/Thread.hpp"
#include "lifuren/Message.hpp"

#include "FL/Fl.H"
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

static void task_finish(void* voidPtr);

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
    const auto iter = thread_worker.find(type);
    if(iter == thread_worker.end()) {
        return;
    }
    if(iter->second->source) {
        static_cast<ThreadWindow*>(iter->second->source)->show();
    }
}

bool lifuren::ThreadWindow::checkThread(lifuren::message::Type type) {
    if(hasThread(type)) {
        showThread(type);
        return false;
    }
    return true;
}

bool lifuren::ThreadWindow::startThread(lifuren::message::Type type, const char* title, std::function<void()> task, std::function<void()> callback) {
    if(!checkThread(type)) {
        return false;
    }
    ThreadWindow* window = new ThreadWindow(800, 600, title);
    window->callback([](Fl_Widget*, void* voidPtr) {
        auto this_window = static_cast<ThreadWindow*>(voidPtr);
        this_window->hide();
        if(this_window->closeable) {
            delete this_window;
        }
    }, window);
    auto worker = std::make_shared<lifuren::thread::ThreadWorker>();
    thread_worker.emplace(type, worker);
    worker->stop   = false;
    worker->type   = lifuren::thread::Type::FLTK;
    worker->source = window;
    worker->thread = std::make_shared<std::thread>([type, task, worker, title, window, callback]() {
        SPDLOG_DEBUG("任务开始：{}", title);
        lifuren::thread::ThreadWorker::this_thread_worker = worker.get();
        try {
            task();
        } catch(const std::exception& e) {
            SPDLOG_ERROR("任务执行异常：{} - {}", title, e.what());
        }
        lifuren::thread::ThreadWorker::this_thread_worker = nullptr;
        SPDLOG_DEBUG("任务完成：{}", title);
        window->closeable = true;
        Fl::awake(task_finish, window);
        if(callback) {
            callback();
        }
        thread_worker.erase(type);
    });
    worker->thread->detach();
    window->show();
    return true;
}

bool lifuren::ThreadWindow::stopThread(lifuren::message::Type type) {
    const auto iter = thread_worker.find(type);
    if(iter == thread_worker.end()) {
        return false;
    }
    iter->second->stop = true;
    return true;
}

static void task_finish(void* voidPtr) {
    static_cast<lifuren::ThreadWindow*>(voidPtr)->show();
}

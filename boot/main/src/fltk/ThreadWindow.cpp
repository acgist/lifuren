#include "lifuren/FLTK.hpp"

#include <map>
#include <memory>
#include <thread>
#include <vector>
#include <algorithm>

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
static std::map<lifuren::message::Type, std::vector<lifuren::message::Type>> depend_tree{
    
};

static void task_finish(void* voidPtr);

lifuren::ThreadWindow::ThreadWindow(int width, int height, const char* title) : Window(width, height, title) {
}

lifuren::ThreadWindow::~ThreadWindow() {
    // 取消回调
    lifuren::message::unregisterMessageCallback(this->type);
    // 释放资源
    LFR_DELETE_PTR(cancel);
    LFR_DELETE_PTR(display);
    LFR_DELETE_PTR(buffer);
}

void lifuren::ThreadWindow::drawElement() {
    // 绘制界面
    display = new Fl_Text_Display(10, 20, this->w() - 20, this->h() - 90, "任务消息");
    buffer  = new Fl_Text_Buffer();
    display->begin();
    display->buffer(buffer);
    display->wrap_mode(display->WRAP_AT_COLUMN, display->textfont());
    display->end();
    cancel = new Fl_Button((this->w() - 200) / 2, this->h() - 50, 200, 30, "取消任务");
    // 注册回调
    lifuren::message::registerMessageCallback(this->type, [](bool finish, const char* message) {
        if(buffer) {
            buffer->append(message);
            buffer->append("\n");
            if(finish) {
                buffer->append("任务完成");
                buffer->append("\n");
            }
        }
    });
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
    return thread_worker.contains(type);
}

bool lifuren::ThreadWindow::startThread(lifuren::message::Type type, const char* title, std::function<void()> task, std::function<void()> callback) {
    if(checkThread(type)) {
        showThread(type);
        return false;
    }
    ThreadWindow* window = new ThreadWindow(LFR_WINDOW_WIDTH / 2, LFR_WINDOW_HEIGHT, title);
    window->type = type;
    window->callback([](Fl_Widget*, void* voidPtr) {
        auto this_window = static_cast<ThreadWindow*>(voidPtr);
        this_window->hide();
        if(this_window->closeable) {
            delete this_window;
        }
    }, window);
    auto worker = std::make_shared<lifuren::thread::ThreadWorker>();
    window->init();
    thread_worker.emplace(type, worker);
    worker->stop   = false;
    worker->type   = lifuren::thread::Type::FLTK;
    worker->source = window;
    worker->thread = std::make_shared<std::thread>([type, task, worker, title, window, callback]() {
        lifuren::message::thread_message_type = type;
        SPDLOG_DEBUG("任务开始：{}", title);
        lifuren::message::sendMessage("任务开始");
        lifuren::thread::ThreadWorker::this_thread_worker = worker.get();
        try {
            task();
        } catch(const std::exception& e) {
            SPDLOG_ERROR("任务执行异常：{} - {}", title, e.what());
        }
        lifuren::thread::ThreadWorker::this_thread_worker = nullptr;
        lifuren::message::sendMessage("任务完成");
        SPDLOG_DEBUG("任务完成：{}", title);
        window->closeable = true;
        Fl::awake(task_finish, window);
        if(callback) {
            callback();
        }
        thread_worker.erase(type);
        lifuren::message::thread_message_type = lifuren::message::Type::NONE;
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

bool lifuren::ThreadWindow::checkDepend(lifuren::message::Type type) {
    auto iterator = depend_tree.find(type);
    if(iterator == depend_tree.end()) {
        return true;
    }
    const auto& depends = iterator->second;
    return std::all_of(depends.begin(), depends.end(), [](const auto& depend) {
        return !checkDepend(depend);
    });
}

bool lifuren::ThreadWindow::checkAudioThread() {
    return std::all_of(thread_worker.begin(), thread_worker.end(), [](const auto& entry) {
        const int value = static_cast<int>(entry.first);
        return value >= MESSAGE_AUDIO_MIN && value < MESSAGE_IMAGE_MIN;
    });
}

bool lifuren::ThreadWindow::checkImageThread() {
    return std::all_of(thread_worker.begin(), thread_worker.end(), [](const auto& entry) {
        const int value = static_cast<int>(entry.first);
        return value >= MESSAGE_IMAGE_MIN && value < MESSAGE_VIDEO_MIN;
    });
}

bool lifuren::ThreadWindow::checkVideoThread() {
    return std::all_of(thread_worker.begin(), thread_worker.end(), [](const auto& entry) {
        const int value = static_cast<int>(entry.first);
        return value >= MESSAGE_VIDEO_MIN && value < MESSAGE_POETRY_MIN;
    });
}

bool lifuren::ThreadWindow::checkPoetryThread() {
    return std::all_of(thread_worker.begin(), thread_worker.end(), [](const auto& entry) {
        const int value = static_cast<int>(entry.first);
        return value >= MESSAGE_POETRY_MIN && value < MESSAGE_MAX;
    });
}

static void task_finish(void* voidPtr) {
    static_cast<lifuren::ThreadWindow*>(voidPtr)->show();
}
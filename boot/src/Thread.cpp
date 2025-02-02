#include "lifuren/Thread.hpp"

thread_local lifuren::thread::ThreadWorker* lifuren::thread::ThreadWorker::this_thread_worker = nullptr;

bool lifuren::thread::ThreadWorker::is_running() {
    return !lifuren::thread::ThreadWorker::this_thread_worker->stop;
}

bool lifuren::thread::ThreadWorker::is_cli_thread() {
    return lifuren::thread::ThreadWorker::this_thread_worker->type == lifuren::thread::Type::CLI;
}

bool lifuren::thread::ThreadWorker::is_fltk_thread() {
    return lifuren::thread::ThreadWorker::this_thread_worker->type == lifuren::thread::Type::FLTK;
}

bool lifuren::thread::ThreadWorker::is_rest_thread() {
    return lifuren::thread::ThreadWorker::this_thread_worker->type == lifuren::thread::Type::REST;
}

lifuren::thread::ThreadTimer::~ThreadTimer() {
    this->shutdown();
}

void lifuren::thread::ThreadTimer::schedule(int interval, std::function<void()> function) {
    std::thread thread([this, interval, function]() {
        while(!this->stop) {
            std::unique_lock<std::mutex> lock(this->mutex);
            this->condition.wait_for(lock, std::chrono::seconds(interval));
            if(this->stop) {
                this->condition.notify_one();
            } else {
                function();
            }
        }
    });
    thread.detach();
}

void lifuren::thread::ThreadTimer::shutdown() {
    this->stop = true;
    this->condition.notify_one();
}

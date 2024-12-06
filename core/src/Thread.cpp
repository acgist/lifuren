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
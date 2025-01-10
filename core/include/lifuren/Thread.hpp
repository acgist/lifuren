/**
 * 多线程工具
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CORE_THREAD_HPP
#define LFR_HEADER_CORE_THREAD_HPP

#include <queue>
#include <mutex>
#include <atomic>
#include <future>
#include <memory>
#include <thread>
#include <vector>
#include <stdexcept>
#include <functional>
#include <condition_variable>

#include "lifuren/Message.hpp"

namespace lifuren::thread {

enum class Type {

    CLI,
    FLTK,
    REST,
    NONE

};

class ThreadWorker {

public:
    // 当前执行线程
    thread_local static ThreadWorker* this_thread_worker;
    // 是否停止
    bool stop { true };
    // 线程类型
    Type type { Type::NONE };
    // 线程来源
    void* source { nullptr };
    // 执行线程
    std::shared_ptr<std::thread> thread{ nullptr };

public:
    ~ThreadWorker();

public:
    // 是否运行
    static bool is_running();
    static bool is_cli_thread();
    static bool is_fltk_thread();
    static bool is_rest_thread();

};

/**
 * 线程池
 */
class ThreadPool {

public:
    ThreadPool(bool bindThreadLocal = true, size_t size = std::thread::hardware_concurrency());
    ~ThreadPool();

public:
    /**
     * 添加任务
     * 
     * @param func 任务
     * @param args 参数
     */
    template<class F, class... Args>
    auto submit(F&& func, Args&&... args) -> std::future<typename std::invoke_result<F, Args...>::type>;
    // 等待完成
    void wait_finish();
    // 唤醒完成
    void notify_finish();

private:
    bool stop;        // 是否关闭
    std::atomic<int> tasks_count{ 0 }; // 任务计数
    std::mutex mutex; // 任务锁
    std::mutex mutex_finish; // 完成锁
    std::condition_variable           condition; // 任务锁条件
    std::condition_variable           condition_finish; // 完成锁锁条件
    std::vector<std::thread>          workers;   // 工作线程
    std::queue<std::function<void()>> tasks;     // 任务队列
};
 
inline ThreadPool::ThreadPool(bool bindThreadLocal, size_t threads) : stop(false) {
    lifuren::message::Type         type               = lifuren::message::Type::NONE;
    lifuren::thread::ThreadWorker* this_thread_worker = nullptr;
    if(bindThreadLocal) {
        type               = lifuren::message::thread_message_type;
        this_thread_worker = lifuren::thread::ThreadWorker::this_thread_worker;
    }
    for(size_t i = 0; i < threads; ++i) {
        this->workers.emplace_back([this, type, this_thread_worker] {
            lifuren::message::thread_message_type             = type;
            lifuren::thread::ThreadWorker::this_thread_worker = this_thread_worker;
            while(true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(this->mutex);
                    this->condition.wait(lock, [this]{ return this->stop || !this->tasks.empty(); });
                    if(this->stop && this->tasks.empty()) {
                        return;
                    }
                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                }
                task();
                --this->tasks_count;
                this->condition_finish.notify_one();
            }
        });
    }
}

template<class F, class... Args>
auto ThreadPool::submit(F&& func, Args&&... args) -> std::future<typename std::invoke_result<F, Args...>::type> {
    using return_type = typename std::invoke_result<F, Args...>::type;
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(func), std::forward<Args>(args)...)
    );
    std::future<return_type> future = task->get_future();
    {
        std::unique_lock<std::mutex> lock(this->mutex);
        if(this->stop) {
            throw std::runtime_error("线程池已关闭");
        }
        // std::move(task)
        this->tasks.emplace([task](){ (*task)(); });
        ++this->tasks_count;
    }
    this->condition.notify_one();
    return future;
}

inline void ThreadPool::wait_finish() {
    std::unique_lock<std::mutex> lock(this->mutex_finish);
    this->condition_finish.wait(lock, [this]() {
        if(lifuren::thread::ThreadWorker::this_thread_worker && lifuren::thread::ThreadWorker::this_thread_worker->stop) {
            return true;
        }
        return this->tasks_count <= 0;
    });
}

inline void ThreadPool::notify_finish() {
    this->condition_finish.notify_one();
}

inline ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(this->mutex);
        this->stop = true;
    }
    this->condition.notify_all();
    for(std::thread& worker: this->workers) {
        worker.join();
    }
}

/**
 * 定时器
 */
class Timer {
    // TODO: impl
};

} // END lifuren

#endif // LFR_HEADER_CORE_THREAD_HPP

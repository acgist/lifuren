/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * 多线程工具
 * 
 * @author acgist
 * 
 * @version 1.0.0
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

/**
 * 线程类型
 */
enum class Type {

    CLI,
    FLTK,
    REST,
    NONE

};

/**
 * 工作线程
 */
class ThreadWorker {

public:
    thread_local static ThreadWorker* this_thread_worker; // 当前执行线程

public:
    bool stop    { true       }; // 是否停止
    Type type    { Type::NONE }; // 线程类型
    void* source { nullptr    }; // 线程来源
    std::shared_ptr<std::thread> thread{ nullptr }; // 执行线程
    // ~ this->thread.join();

public:
    static bool is_running();     // 是否运行
    static bool is_cli_thread();  // 是否CLI线程
    static bool is_fltk_thread(); // 是否FLTK线程
    static bool is_rest_thread(); // 是否REST线程

};

/**
 * 线程池
 */
class ThreadPool {

public:
    ThreadPool(
        bool bindThreadLocal = true, // 是否绑定线程
        size_t size = std::thread::hardware_concurrency() // 线程数量
    );
    ~ThreadPool();

public:
    /**
     * 添加任务
     * 
     * @return 执行结果
     */
    template<class F, class... Args>
    auto submit(
        F   &&    func, // 任务
        Args&&... args  // 参数
    ) -> std::future<typename std::invoke_result<F, Args...>::type>;
    /**
     * 等待完成
     */
    void wait_finish();
    /**
     * 唤醒完成
     */
    void notify_finish();

private:
    bool stop; // 是否关闭
    std::atomic<int> tasks_count{ 0 }; // 任务计数
    std::mutex mutex;        // 任务锁
    std::mutex mutex_finish; // 完成锁
    std::condition_variable  condition;        // 任务锁条件
    std::condition_variable  condition_finish; // 完成锁锁条件
    std::vector<std::thread> workers; // 工作线程
    std::queue<std::function<void()>> tasks; // 任务队列
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
                    this->condition.wait(lock, [this] {
                        return this->stop || !this->tasks.empty();
                    });
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
        this->tasks.emplace([task]() {
            (*task)();
        });
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
class ThreadTimer {

private:
    bool stop = false;
    std::mutex mutex;
    std::condition_variable condition;

public:
    ~ThreadTimer();

public:
    /**
     * 定时任务
     */
    void schedule(
        int interval, // 间隔
        std::function<void()> function // 任务
    );
    void shutdown(); // 结束任务

};

} // END lifuren::thread

#endif // LFR_HEADER_CORE_THREAD_HPP

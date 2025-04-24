/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * 多线程
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_BOOT_THREAD_HPP
#define LFR_HEADER_BOOT_THREAD_HPP

#include <queue>
#include <mutex>
#include <atomic>
#include <future>
#include <thread>
#include <vector>
#include <functional>
#include <condition_variable>

namespace lifuren::thread {

/**
 * 线程池
 */
class ThreadPool {

using thread_task = std::function<void()>;

public:
    /**
     * @param threads 线程数量
     */
    ThreadPool(size_t threads = std::thread::hardware_concurrency());
    ~ThreadPool();

public:
    /**
     * 添加任务
     * 
     * @param func 任务
     * @param args 参数
     * 
     * @return 执行结果
     */
    template<class F, class... Args>
    auto submit(F&& func, Args&&... args) -> std::future<typename std::invoke_result<F, Args...>::type>;
    /**
     * 等待完成
     */
    void awaitTermination();

private:
    bool stop; // 是否关闭
    std::atomic<int> tasks_size{ 0 }; // 任务计数
    std::mutex mutex_t; // 任务锁
    std::mutex mutex_w; // 等待锁
    std::condition_variable condition_t; // 任务锁条件
    std::condition_variable condition_w; // 等待锁条件
    std::queue<thread_task>  tasks;   // 任务队列
    std::vector<std::thread> workers; // 工作线程
};
 
inline ThreadPool::ThreadPool(size_t threads) : stop(false) {
    for(size_t i = 0; i < threads; ++i) {
        this->workers.emplace_back([this] {
            while(true) {
                thread_task task{ nullptr };
                {
                    std::unique_lock<std::mutex> lock(this->mutex_t);
                    this->condition_t.wait(lock, [this] {
                        return this->stop || !this->tasks.empty();
                    });
                    if(this->stop && this->tasks.empty()) {
                        return;
                    }
                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                }
                task();
                --this->tasks_size;
                this->condition_w.notify_one();
            }
        });
    }
}

template<class F, class... Args>
auto ThreadPool::submit(F&& func, Args&&... args) -> std::future<typename std::invoke_result<F, Args...>::type> {
    using result = typename std::invoke_result<F, Args...>::type;
    auto task = std::make_shared<std::packaged_task<result()>>(std::bind(std::forward<F>(func), std::forward<Args>(args)...));
    std::future<result> future = task->get_future();
    {
        std::unique_lock<std::mutex> lock(this->mutex_t);
        if(this->stop) {
            throw std::runtime_error("线程池已关闭");
        }
        this->tasks.emplace([task]() {
            (*task)();
        });
        ++this->tasks_size;
    }
    this->condition_t.notify_one();
    return future;
}

inline void ThreadPool::awaitTermination() {
    std::unique_lock<std::mutex> lock(this->mutex_w);
    this->condition_w.wait(lock, [this]() {
        // 注意：不能直接判断队列长度
        return this->tasks_size <= 0;
    });
}

inline ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(this->mutex_t);
        this->stop = true;
    }
    this->condition_t.notify_all();
    for(std::thread& worker: this->workers) {
        worker.join();
    }
}

} // END lifuren::thread

#endif // LFR_HEADER_BOOT_THREAD_HPP

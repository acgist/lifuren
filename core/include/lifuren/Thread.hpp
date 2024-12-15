/**
 * 多线程工具
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CORE_THREAD_HPP
#define LFR_HEADER_CORE_THREAD_HPP

#include <queue>
#include <mutex>
#include <future>
#include <memory>
#include <thread>
#include <vector>
#include <stdexcept>
#include <functional>
#include <condition_variable>

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
    bool stop{ true };
    // 线程类型
    Type type { Type::NONE };
    // 线程来源
    void* source { nullptr };
    // 执行线程
    std::shared_ptr<std::thread> thread{ nullptr };

public:
    // 是否运行
    static bool is_running();
    static bool is_cli_thread();
    static bool is_fltk_thread();
    static bool is_rest_thread();

};

/**
 * 线程池
 * 
 * https://github.com/progschj/ThreadPool
 */
class ThreadPool {

public:
    ThreadPool(size_t);
    template<typename F, typename... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::invoke_result<F(Args...)>::type>;
    ~ThreadPool();

private:
    bool stop;
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;

};
 
inline ThreadPool::ThreadPool(size_t threads) : stop(false) {
    for(size_t i = 0; i < threads; ++i) {
        workers.emplace_back(
            [this] {
                for(;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
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
                }
            }
        );
    }
}

template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args)  -> std::future<typename std::invoke_result<F(Args...)>::type> {
    using return_type = typename std::invoke_result<F(Args...)>::type;
    auto task = std::make_shared<std::packaged_task<return_type()>>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        if(stop) {
            throw std::runtime_error("enqueue on stopped ThreadPool");
        }
        tasks.emplace([task]() {
            (*task)();
        });
    }
    condition.notify_one();
    return res;
}

inline ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    for(std::thread& worker: workers) {
        worker.join();
    }
}

} // END lifuren

#endif // LFR_HEADER_CORE_THREAD_HPP

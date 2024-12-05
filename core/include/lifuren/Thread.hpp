/**
 * 多线程工具
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CORE_THREAD_HPP
#define LFR_HEADER_CORE_THREAD_HPP

#include <memory>
#include <thread>

namespace lifuren::thread {

class ThreadWorker {

public:
    // 是否FLTK线程
    thread_local static bool fltk_thread;
    // 当前执行线程
    thread_local static ThreadWorker* this_thread_worker;
    // 是否停止
    bool stop{ true };
    // 执行线程
    std::shared_ptr<std::thread> thread{ nullptr };

};

thread_local bool          ThreadWorker::fltk_thread        = false;
thread_local ThreadWorker* ThreadWorker::this_thread_worker = nullptr;

} // END lifuren

#endif // LFR_HEADER_CORE_THREAD_HPP

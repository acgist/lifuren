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

} // END lifuren

#endif // LFR_HEADER_CORE_THREAD_HPP

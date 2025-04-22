/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * 测试
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_TEST_HPP
#define LFR_HEADER_TEST_HPP

#include <chrono>
#include <thread>
#include <cassert>
#include <cstdlib>
#include <cstdint>
#include <functional>

#include "spdlog/spdlog.h"

#include "lifuren/Config.hpp"
#include "lifuren/Logger.hpp"

#ifndef LFR_TEST
#define LFR_TEST(...)                                \
int main(const int argc, const char* const argv[]) { \
    lifuren::logger::init();                         \
    lifuren::config::init(argc, argv);               \
    try {                                            \
        __VA_ARGS__                                  \
    } catch(const std::exception& e) {               \
        SPDLOG_ERROR("{}", e.what());                \
    }                                                \
    lifuren::logger::stop();                         \
    return 0;                                        \
}
#endif

namespace lifuren::test {

/**
 * 循环执行
 * 
 * @param count     循环次数
 * @param function  循环函数
 * @param loop_wait 循环等待时间
 * @param last_wait 最后等待时间
 * @param lable     提示内容
 * 
 * @returns 消耗时间（毫秒）
 */
inline size_t loop(size_t count, std::function<void()> function, const int loop_wait = 0, const int last_wait = 0, const char* label = "消耗时间") {
    const auto a = std::chrono::system_clock::now();
    for(size_t i = 0; i < count; ++i) {
        function();
        if(loop_wait > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(loop_wait));
        }
    }
    const auto z = std::chrono::system_clock::now();
    size_t ret = std::chrono::duration_cast<std::chrono::milliseconds>(z - a).count();
    SPDLOG_INFO("{}：{}", label, ret);
    if(last_wait > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(last_wait));
    } 
    return ret;
}

} // END OF lifuren::test

#endif // END OF LFR_HEADER_TEST_HPP

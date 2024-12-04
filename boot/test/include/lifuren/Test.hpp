/**
 * 测试
 */
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
#define LFR_TEST(...)                                 \
int main(const int argc, const char* const argv[]) {  \
    lifuren::logger::init();                          \
    lifuren::config::init(argc, argv);                \
    try {                                             \
        __VA_ARGS__                                   \
    } catch(const std::exception& e) {                \
        SPDLOG_ERROR("{}", e.what());                 \
    }                                                 \
    lifuren::logger::shutdown();                      \
    return 0;                                         \
}
#endif

#ifndef LFR_MEM_TEST
#define LFR_MEM_TEST(...)                                                 \
int main(const int argc, const char* const argv[]) {                      \
    lifuren::logger::init();                                              \
    lifuren::config::init(argc, argv);                                    \
    try {                                                                 \
        while(true) {                                                     \
            __VA_ARGS__                                                   \
            std::this_thread::sleep_for(std::chrono::milliseconds(1000)); \
        }                                                                 \
    } catch(const std::exception& e) {                                    \
        SPDLOG_ERROR("{}", e.what());                                     \
    }                                                                     \
    lifuren::logger::shutdown();                                          \
    return 0;                                                             \
}
#endif

/**
 * @param count    循环次数
 * @param function 循环函数
 * @param label    提示内容
 * 
 * @returns 消耗时间（毫秒）
 */
inline size_t cost(size_t count, std::function<void()> function, const char* label = "消耗时间") {
    const auto a = std::chrono::system_clock::now();
    for(size_t i = 0; i < count; ++i) {
        function();
    }
    const auto z = std::chrono::system_clock::now();
    size_t ret = std::chrono::duration_cast<std::chrono::milliseconds>(z - a).count();
    SPDLOG_DEBUG("{}：{}", label, ret);
    return ret;
}
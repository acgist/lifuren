/**
 * 测试
 */
#include <cassert>

#include "spdlog/spdlog.h"

#include "lifuren/Config.hpp"
#include "lifuren/Logger.hpp"

#ifndef LFR_TEST
#define LFR_TEST(...)                                 \
int main(const int argc, const char* const argv[]) {  \
    lifuren::logger::init();                          \
    lifuren::config::initBase(argc, argv);            \
    __VA_ARGS__                                       \
    lifuren::logger::shutdown();                      \
    return 0;                                         \
}
#endif

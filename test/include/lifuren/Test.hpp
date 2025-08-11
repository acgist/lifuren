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

#include "spdlog/spdlog.h"

#include "lifuren/Config.hpp"
#include "lifuren/Logger.hpp"

#ifdef _WIN32
const char* tmp_directory = "D:/tmp";
#else
const char* tmp_directory = "/tmp";
#endif

#ifndef LFR_TEST
#define LFR_TEST(...)                                \
int main(const int argc, const char* const argv[]) { \
    lifuren::logger::init();                         \
    lifuren::logger::opencv::init();                 \
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

#endif // END OF LFR_HEADER_TEST_HPP

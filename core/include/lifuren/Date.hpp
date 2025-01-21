/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * 日期工具
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_CORE_DATE_HPP
#define LFR_HEADER_CORE_DATE_HPP

#include <chrono>
#include <string>

#include <time.h>

#ifndef LFR_DATE_TIME_FORMAT
#define LFR_DATE_TIME_FORMAT "%Y-%m-%d %H:%M:%S"
#endif

namespace lifuren::date {

/**
 * @return 格式日期时间
 */
extern std::string format(
    const std::tm    & datetime, // 日期时间
    const std::string& format    // 格式
);

/**
 * @return 格式日期时间
 */
extern std::string format(
    const std::chrono::system_clock::time_point& datetime, // 日期时间
    const std::string& format // 格式
);

/**
 * @return 日期时间
 */
extern std::tm parse_tm(
    const std::string& datetime, // 格式日期时间
    const std::string& format    // 格式
);

/**
 * @return 日期时间
 */
extern std::chrono::system_clock::time_point parse_time_point(
    const std::string& datetime, // 格式日期时间
    const std::string& format    // 格式
);

/**
 * @return 毫秒
 */
extern uint64_t to_millis(
    std::tm& datetime // 日期时间
);

/**
 * @return 毫秒
 */
extern uint64_t to_millis(
    const std::chrono::system_clock::time_point& datetime // 日期时间
);

/**
 * @return 日期时间
 */
extern std::tm parse_tm(
    const uint64_t& millis // 毫秒
);

/**
 * @return 日期时间
 */
extern std::chrono::system_clock::time_point parse_time_point(
    const uint64_t& millis // 毫秒
);

/**
 * 设置时区
 */
inline void setTimeZone(
    const char* timezone = "Asia/Shanghai" // 时区
) {
    #ifdef _WIN32
    _putenv_s("TZ", timezone);
    _tzset();
    #elif defined(__linux) || defined(__linux__)
    setenv("TZ", timezone, true);
    tzset();
    #else
    // -
    #endif
}

} // END OF lifuren::data

#endif // LFR_HEADER_CORE_DATE_HPP

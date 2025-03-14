/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * 日期
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_BOOT_DATE_HPP
#define LFR_HEADER_BOOT_DATE_HPP

#include <chrono>
#include <string>

#include <time.h>

#ifndef LFR_DATE_TIME_FORMAT
#define LFR_DATE_TIME_FORMAT "%Y-%m-%d %H:%M:%S"
#endif

namespace lifuren::date {

/**
 * @param datetime 日期时间
 * @param format   格式
 * 
 * @return 格式日期时间
 */
extern std::string format(const std::tm& datetime, const std::string& format);

/**
 * @param datetime 日期时间
 * @param format   格式
 * 
 * @return 格式日期时间
 */
extern std::string format(const std::chrono::system_clock::time_point& datetime, const std::string& format);

/**
 * @param datetime 格式日期时间
 * @param format   格式
 * 
 * @return 日期时间
 */
extern std::tm parse_tm(const std::string& datetime, const std::string& format);

/**
 * @param datetime 格式日期时间
 * @param format   格式
 * 
 * @return 日期时间
 */
extern std::chrono::system_clock::time_point parse_time_point(const std::string& datetime, const std::string& format);

/**
 * @param datetime 日期时间
 * 
 * @return 毫秒
 */
extern uint64_t to_millis(std::tm& datetime);

/**
 * @param datetime 日期时间
 * 
 * @return 毫秒
 */
extern uint64_t to_millis(const std::chrono::system_clock::time_point& datetime);

/**
 * @param millis 毫秒
 * 
 * @return 日期时间
 */
extern std::tm parse_tm(uint64_t millis);

/**
 * @param millis 毫秒
 * 
 * @return 日期时间
 */
extern std::chrono::system_clock::time_point parse_time_point(uint64_t millis);

/**
 * @param timezone 时区
 */
inline void setTimeZone(const char* timezone = "Asia/Shanghai") {
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

#endif // LFR_HEADER_BOOT_DATE_HPP

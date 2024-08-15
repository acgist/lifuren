/**
 * 日期工具
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CORE_DATES_HPP
#define LFR_HEADER_CORE_DATES_HPP

#include <chrono>
#include <string>

#include <time.h>

#ifndef LFR_DATE_TIME_FORMAT
#define LFR_DATE_TIME_FORMAT "%Y-%m-%d %H:%M:%S"
#endif

namespace lifuren {
namespace dates   {

/**
 * @param datetime 日期时间
 * @param format   格式
 * 
 * @return 格式化后日期时间
 */
extern std::string format(const std::tm& datetime, const std::string& format);

/**
 * @param datetime 日期时间
 * @param format   格式
 * 
 * @return 格式化后日期时间
 */
extern std::string format(const std::chrono::system_clock::time_point& datetime, const std::string& format);

/**
 * @param datetime 格式化后日期时间
 * @param format   格式
 * 
 * @return 日期时间
 */
extern std::tm parseTm(const std::string& datetime, const std::string& format);

/**
 * @param datetime 格式化后日期时间
 * @param format   格式
 * 
 * @return 日期时间
 */
extern std::chrono::system_clock::time_point parseTp(const std::string& datetime, const std::string& format);

/**
 * @param datetime 日期时间
 * 
 * @return 毫秒
 */
extern uint64_t toMillis(std::tm& datetime);

/**
 * @param datetime 日期时间
 * 
 * @return 毫秒
 */
extern uint64_t toMillis(const std::chrono::system_clock::time_point& datetime);

/**
 * @param millis 毫秒
 * 
 * @return 日期时间
 */
extern std::tm parseTm(const uint64_t& millis);

/**
 * @param millis 毫秒
 * 
 * @return 日期时间
 */
extern std::chrono::system_clock::time_point parseTp(const uint64_t& millis);

/**
 * 设置时区
 * 
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
    // 其他
    #endif
}

}
}

#endif // LFR_HEADER_CORE_DATES_HPP

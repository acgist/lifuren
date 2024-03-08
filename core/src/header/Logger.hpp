/**
 * 日志工具
 * 
 * st：单线程版本（效率更高）
 * mt：多线程版本（线程安全）
 * 
 * @author acgist
 */
#pragma once

#include "spdlog/spdlog.h"
#include "spdlog/fmt/ostr.h"
#include "spdlog/fmt/chrono.h"
#include "spdlog/fmt/ranges.h"
#include "spdlog/sinks/daily_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#ifndef LFR_LOG_FORMAT
#define LFR_LOG_FORMAT(type)                      \
template<>                                        \
struct fmt::formatter<type> : ostream_formatter { \
};
#endif

template<typename T>
typename std::enable_if<std::is_enum<T>::value, std::ostream>::type& operator<<(std::ostream& out, const T& v) {
    return out << static_cast<int>(v);
}

namespace lifuren {
namespace logger  {

/**
 * 加载日志
 */
extern void init();

/**
 * 关闭日志
 */
extern void shutdown();

}
}

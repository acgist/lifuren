/**
 * 日志工具
 * 
 * st：单线程版本（效率更高）
 * mt：多线程版本（线程安全）
 * 
 * @author acgist
 */
#pragma once

#include "fmt/chrono.h"
#include "fmt/ranges.h"
#include "fmt/ostream.h"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/daily_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "opencv2/core/utils/logger.hpp"

#ifndef LFR_LOG_FORMAT_ENUM
#define LFR_LOG_FORMAT_ENUM(type)      \
inline auto format_as(const type& t) { \
    return fmt::underlying(t);         \
}
#endif

#ifndef LFR_LOG_FORMAT_STREAM
#define LFR_LOG_FORMAT_STREAM(type)               \
template<>                                        \
struct fmt::formatter<type> : ostream_formatter { \
};
#endif

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

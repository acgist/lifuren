/**
 * 日志工具
 * 
 * st：单线程版本（效率更高）
 * mt：多线程版本（线程安全）
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CORE_LOGGER_HPP
#define LFR_HEADER_CORE_LOGGER_HPP

// 定义日志枚举翻译
#ifndef LFR_LOG_FORMAT_ENUM
#define LFR_LOG_FORMAT_ENUM(type)      \
inline auto format_as(const type& t) { \
    return fmt::underlying(t);         \
}
#endif

// 定义日志流的翻译
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

#endif // LFR_HEADER_CORE_LOGGER_HPP

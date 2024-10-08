/**
 * 日志工具
 * 
 * st：单线程版本（效率更高）
 * mt：多线程版本（线程安全）
 * 
 * https://fmt.dev/11.0/
 * https://fmt.dev/11.0/syntax/
 * 
 * 
 * replacement_field ::= "{" [arg_id] [":" (format_spec | chrono_format_spec)] "}"
 * 
 * arg_id            ::= integer | identifier
 * 
 * format_spec ::= [[fill]align][sign]["#"]["0"][width]["." precision]["L"][type]
 * fill        ::= <a character other than '{' or '}'>
 * align       ::= "<" | ">" | "^"
 * sign        ::= "+" | "-" | " "
 * width       ::= integer | "{" [arg_id] "}"
 * precision   ::= integer | "{" [arg_id] "}"
 * type        ::= "a" | "A" | "b" | "B" | "c" | "d" | "e" | "E" | "f" | "F" |
 *                 "g" | "G" | "o" | "p" | "s" | "x" | "X" | "?"
 * 
 * chrono_format_spec ::= [[fill]align][width]["." precision][chrono_specs]
 * chrono_specs       ::= conversion_spec | chrono_specs (conversion_spec | literal_char)
 * conversion_spec    ::= "%" [padding_modifier] [locale_modifier] chrono_type
 * literal_char       ::= <a character other than '{', '}' or '%'>
 * padding_modifier   ::= "-" | "_"  | "0"
 * locale_modifier    ::= "E" | "O"
 * chrono_type        ::= "a" | "A" | "b" | "B" | "c" | "C" | "d" | "D" | "e" |
 *                        "F" | "g" | "G" | "h" | "H" | "I" | "j" | "m" | "M" |
 *                        "n" | "p" | "q" | "Q" | "r" | "R" | "S" | "t" | "T" |
 *                        "u" | "U" | "V" | "w" | "W" | "x" | "X" | "y" | "Y" |
 *                        "z" | "Z" | "%"
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CORE_LOGGER_HPP
#define LFR_HEADER_CORE_LOGGER_HPP

// 日志枚举翻译
#ifndef LFR_FORMAT_LOG_ENUM
#define LFR_FORMAT_LOG_ENUM(type)      \
inline auto format_as(const type& t) { \
    return fmt::underlying(t);         \
}
#endif

// 日志输出流翻译
#ifndef LFR_FORMAT_LOG_STREAM
#define LFR_FORMAT_LOG_STREAM(type)               \
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

} // END OF logger
} // END OF lifuren

#endif // LFR_HEADER_CORE_LOGGER_HPP

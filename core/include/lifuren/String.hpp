/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * 字符串工具
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_CORE_STRING_HPP
#define LFR_HEADER_CORE_STRING_HPP

#include <regex>
#include <cctype>
#include <string>
#include <vector>
#include <codecvt>
#include <sstream>
#include <algorithm>

namespace lifuren::string {

const char* const EMPTY_CHARS = " \t\r\n"; // 空白字符

/**
 * @return 拼接内容
 */
template<typename T>
std::string join(
    const T          & values,    // 拼接集合
    const std::string& delim = "" // 拼接符号
) {
    if(values.empty()) {
        return {};
    }
          typename T::const_iterator beg  = values.begin();
    const typename T::const_iterator end  = values.end();
    const typename T::const_iterator last = values.end() - 1;
    std::stringstream ret;
    for (; beg != end; ++beg) {
        ret << *beg;
        if (beg != last) {
            ret << delim;
        }
    }
    return ret.str();
}

/**
 * @return 是否数字
 */
inline bool isNumeric(
    const std::string& v // 字符串
) {
    static const std::regex regex(R"(^(\-|\+)?[0-9]+(\.[0-9]+)?$)");
    return std::regex_match(v, regex);
}

/**
 * @return 分割列表
 */
extern std::vector<std::string> split(
    const std::string& content, // 文本内容
    const std::string& delim,   // 分隔符号
    bool retain = false, // 是否保留分隔符号
    bool filter = true   // 是否过滤空白字符
);

/**
 * @return 分割列表
 */
extern std::vector<std::string> split(
    const std::string             & content, // 内容文本
    const std::vector<std::string>& multi,   // 分隔符号列表
    bool retain = false, // 是否保留分隔符号
    bool filter = true   // 是否过滤空白字符
);

/**
 * 转为小写
 */
inline void toLower(
    std::string& value // 字符串
) {
    #if _WIN32
    std::transform(value.begin(), value.end(), value.begin(), ::tolower);
    #else
    std::transform(value.begin(), value.end(), value.begin(), [](const char& v) -> char {
        return std::tolower(v);
    });
    #endif
}

/**
 * 转为大写
 */
inline void toUpper(
    std::string& value // 字符串
) {
    #if _WIN32
    std::transform(value.begin(), value.end(), value.begin(), ::toupper);
    #else
    std::transform(value.begin(), value.end(), value.begin(), [](const char& v) -> char {
        return std::toupper(v);
    });
    #endif
}

/**
 * @return 去掉空格后的字符串
 */
extern char* trim(
    char* value // 字符串
);

/**
 * @return 去掉空格后的字符串
 */
extern std::string trim(
    const std::string& value // 字符串
);

/**
 * @return 字符串长度
 */
extern size_t length(
    const char* value // UTF8字符串
);

/**
 * @return 字符串长度
 */
inline size_t length(
    const std::string& value // UTF8字符串
) {
    return lifuren::string::length(value.c_str());
}

/**
 * 1字节 0xxxxxxx
 * 2字节 110xxxxx 10xxxxxx
 * 3字节 1110xxxx 10xxxxxx 10xxxxxx
 * 4字节 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
 * 
 * @return 真实偏移位置
 */
inline uint32_t indexPos(
    const char    * value, // UTF8字符串
          uint32_t& pos,   // 偏移位置
    const uint32_t& size   // 偏移长度
) {
    uint32_t index = 0;
    if(index < size) {
        while(value[pos]) {
            if((value[pos] & 0xC0) != 0x80) {
                ++index;
            }
            ++pos;
            if((value[pos] & 0xC0) != 0x80) {
                if(index >= size) {
                    break;
                }
            }
        }
    }
    return pos;
}

/**
 * @return 文本
 */
inline std::wstring to_wstring(
    const std::string& input // 文本
) {
	std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
	return converter.from_bytes(input);
}

/**
 * @return 文本
 */
inline std::string to_string(
    const std::wstring& input // 文本
) {
	std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
	return converter.to_bytes(input);
}

/**
 * @return 截取内容
 */
extern std::string substr(
    const char    * value,  // UTF8字符串
    const uint32_t& offset, // 开始位置
    const uint32_t& length  // 截取长度
);

/**
 * @return UTF8字符列表
 */
extern std::vector<std::string> toChars(
    const std::string& segment, // UTF8字符串
    bool filter = true // 是否过滤空白字符
);

/**
 * 字符串替换
 */
extern void replace(
    std::string      & value,        // 字符串
    const std::string& oldValue,     // 旧的字符串
    const std::string& newValue = "" // 新的字符串
);

/**
 * 字符串替换
 */
extern void replace(
    std::string                   & value,        // 字符串
    const std::vector<std::string>& oldValue,     // 旧的字符串列表
    const std::string             & newValue = "" // 新的字符串
);

} // END OF lifuren::string

#endif // LFR_HEADER_CORE_STRING_HPP

/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * 字符串
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_BOOT_STRING_HPP
#define LFR_HEADER_BOOT_STRING_HPP

#include <regex>
#include <cctype>
#include <string>
#include <vector>
#include <codecvt>
#include <sstream>
#include <algorithm>

namespace lifuren::string {

/**
 * @param list  拼接集合
 * @param delim 拼接符号
 * 
 * @return 拼接内容
 */
template<typename T>
std::string join(const T& list, const std::string& delim = "") {
    if(list.empty()) {
        return {};
    }
          typename T::const_iterator beg  = list.begin();
    const typename T::const_iterator end  = list.end();
    const typename T::const_iterator last = list.end() - 1;
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
 * @param content 文本内容
 * @param delim   分割符号
 * @param retail  是否保留分割符号
 * @param filter  是否过滤空白符号
 * 
 * @return 分割列表
 */
extern std::vector<std::string> split(const std::string& content, const std::string& delim, bool retain = false, bool filter = true);

/**
 * @param content 文本内容
 * @param multi   分割符号列表
 * @param retail  是否保留分割符号
 * @param filter  是否过滤空白符号
 * 
 * @return 分割列表
 */
extern std::vector<std::string> split(const std::string& content, const std::vector<std::string>& multi, bool retain = false, bool filter = true);

/**
 * @param value 字符串
 * 
 * @return 是否数字
 */
inline bool isNumeric(const std::string& value) {
    static const std::regex regex(R"(^(\-|\+)?[0-9]+(\.[0-9]+)?$)");
    return std::regex_match(value, regex);
}

/**
 * @param value 字符串
 */
inline void toLower(std::string& value) {
    #if _WIN32
    std::transform(value.begin(), value.end(), value.begin(), ::tolower);
    #else
    std::transform(value.begin(), value.end(), value.begin(), [](const char& v) -> char {
        return std::tolower(v);
    });
    #endif
}

/**
 * @param value 字符串
 */
inline void toUpper(std::string& value) {
    #if _WIN32
    std::transform(value.begin(), value.end(), value.begin(), ::toupper);
    #else
    std::transform(value.begin(), value.end(), value.begin(), [](const char& v) -> char {
        return std::toupper(v);
    });
    #endif
}

/**
 * @param value 字符串
 * 
 * @return 去掉空格后的字符串
 */
extern char* trim(char* value);

/**
 * @param value 字符串
 * 
 * @return 去掉空格后的字符串
 */
extern std::string trim(const std::string& value);

/**
 * @param value UTF-8字符串
 * 
 * @return 字符串长度
 */
extern size_t length(const char* value);

/**
 * @param value UTF-8字符串
 * 
 * @return 字符串长度
 */
inline size_t length(const std::string& value) {
    return lifuren::string::length(value.c_str());
}

/**
 * 1字节 0xxxxxxx
 * 2字节 110xxxxx 10xxxxxx
 * 3字节 1110xxxx 10xxxxxx 10xxxxxx
 * 4字节 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
 * 
 * @param value UTF-8字符串
 * @param pos   偏移位置
 * @param size  偏移长度
 * 
 * @return 真实偏移位置
 */
extern uint32_t indexPos(const char* value, uint32_t& pos, const uint32_t& size);

/**
 * @param input 文本
 * 
 * @return 文本
 */
inline std::wstring to_wstring(const std::string& input) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
	return converter.from_bytes(input);
}

/**
 * @param input 文本
 * 
 * @return 文本
 */
inline std::string to_string(const std::wstring& input) {
	std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
	return converter.to_bytes(input);
}

/**
 * @param value  UTF-8字符串
 * @param offset 开始位置
 * @param length 截取长度
 * 
 * @return 截取内容
 */
extern std::string substr(const char* value, const uint32_t& offset, const uint32_t& length);

/**
 * @param value  UTF-8字符串
 * @param filter 是否过滤空白字符
 * 
 * @return UTF-8字符列表
 */
extern std::vector<std::string> toChars(const std::string& value, bool filter = true);

/**
 * @param value    字符串
 * @param oldValue 旧的字符串
 * @param newValue 新的字符串
 */
extern void replace(std::string& value, const std::string& oldValue, const std::string& newValue = "");

/**
 * @param value    字符串
 * @param oldValue 旧的字符串列表
 * @param newValue 新的字符串
 */
extern void replace(std::string& value, const std::vector<std::string>& oldValue, const std::string& newValue = "");

} // END OF lifuren::string

#endif // LFR_HEADER_BOOT_STRING_HPP

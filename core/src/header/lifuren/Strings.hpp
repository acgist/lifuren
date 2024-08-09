/**
 * 字符串
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CORE_STRINGS_HPP
#define LFR_HEADER_CORE_STRINGS_HPP

#include <string>
#include <vector>

namespace lifuren {
namespace strings {

// 空白符号
const char* const EMPTY_CHARS = " \t\r\n";

/**
 * 转为小写
 * 
 * @param value 字符串
 */
extern void toLower(std::string& value);

/**
 * 转为大写
 * 
 * @param value 字符串
 */
extern void toUpper(std::string& value);

/**
 * @param value 字符串
 * 
 * @return 去掉空格后的字符串
 */
extern std::string trim(const std::string& value);

/**
 * @param value 字符串
 * 
 * @return 去掉空格后的字符串
 */
extern char* trim(char* value);

/**
 * @param value UTF8字符串
 * 
 * @return 字符串长度
 */
extern size_t length(const char* value);

/**
 * @param value UTF8字符串
 * 
 * @return 字符串长度
 */
inline size_t length(const std::string& value) {
    return lifuren::strings::length(value.c_str());
}

/**
 * 1字节 0xxxxxxx
 * 2字节 110xxxxx 10xxxxxx
 * 3字节 1110xxxx 10xxxxxx 10xxxxxx
 * 4字节 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
 * 
 * @param value  UTF8字符串
 * @param pos    开始位置
 * @param length 截取长度
 * 
 * @return 截取内容
 */
extern std::string substr(const char* value, uint32_t& pos, const uint32_t& length);

/**
 * @param value  UTF8字符串
 * @param pos    开始位置
 * @param length 截取长度
 * 
 * @return 截取内容
 */
inline std::string substr(const char* value, const uint32_t& pos, const uint32_t& length) {
    uint32_t copy = pos;
    return substr(value, copy, length);
}

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

}
}

#endif // LFR_HEADER_CORE_STRINGS_HPP

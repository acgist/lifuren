/**
 * 字符串
 * 
 * @author acgist
 */
#pragma once

#include <string>
#include <vector>
#include <algorithm>

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
inline std::string trim(const std::string& value) {
    std::size_t index = value.find_first_not_of(EMPTY_CHARS);
    if(index == std::string::npos) {
        return std::string();
    }
    std::size_t jndex = value.find_last_not_of(EMPTY_CHARS);
    return value.substr(index, jndex + 1 - index);
}

/**
 * @param value 字符串
 * 
 * @return 字符串长度
 */
inline size_t length(const char* value) {
    size_t index = 0;
    size_t jndex = 0;
    while (value[index]) {
        if ((value[index] & 0xC0) != 0x80) {
            ++jndex;
        };
        ++index;
    }
    return jndex;
}

/**
 * @param value 字符串
 * 
 * @return 字符串长度
 */
inline size_t length(const std::string& value) {
    return lifuren::strings::length(value.c_str());
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
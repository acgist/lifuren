/**
 * 字符串
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CORE_STRINGS_HPP
#define LFR_HEADER_CORE_STRINGS_HPP

#include <string>
#include <vector>
#include <sstream>

namespace lifuren {
namespace strings {

// 空白符号
const char* const EMPTY_CHARS = " \t\r\n";

/**
 * @param collection 拼接集合
 * @param delim      分隔符号
 * 
 * @return 拼接内容
 */
template<typename T>
std::string join(T& collection, const std::string& delim = "") {
    std::stringstream ret;
    if(collection.empty()) {
        return ret.str();
    }
    typename T::iterator iterator = collection.begin();
    const typename T::const_iterator end  = collection.end();
    const typename T::const_iterator last = collection.end() - 1;
    for (; iterator != end; ++iterator) {
        ret << *iterator;
        if (iterator != last) {
            ret << delim;
        }
    }
    return ret.str();
}

/**
 * @param content 文本内容
 * @param delim   分隔符号
 * @param retain  保留分割符号
 * @param filter  是否过滤空白字符
 * 
 * @return 分割列表
 */
extern std::vector<std::string> split(const std::string& content, const std::string& delim, bool retain = false, bool filter = true);

/**
 * @param content 文本内容
 * @param multi   分隔符号列表
 * @param retain  保留分割符号
 * @param filter  是否过滤空白字符
 * 
 * @return 分割列表
 */
extern std::vector<std::string> split(const std::string& content, const std::vector<std::string>& multi, bool retain = false, bool filter = true);

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
 * @param segment 字符串
 * @param filter  是否过滤空白字符
 */
extern std::vector<std::string> toChars(const std::string& segment, bool filter = true);

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

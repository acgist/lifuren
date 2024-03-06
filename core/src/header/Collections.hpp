/**
 * 集合
 * 
 * @author acgist
 */
#pragma once

#include <map>
#include <list>
#include <vector>
#include <string>
#include <sstream>

namespace lifuren {
namespace collections {

/**
 * @param collection 集合
 * @param delim      分隔符号
 * 
 * @return 拼接内容
 */
template <typename T>
std::string join(T& collection, const std::string& delim) {
    std::stringstream ret;
    if(collection.empty()) {
        return ret.str();
    }
    std::string x;
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
 * 
 * @return 分割列表
 */
extern std::vector<std::string> split(const std::string& content, const std::string& delim);

/**
 * @param content 文本内容
 * @param multi   分隔符号
 * 
 * @return 分割列表
 */
extern std::vector<std::string> split(const std::string& content, const std::vector<std::string>& multi);

}
}

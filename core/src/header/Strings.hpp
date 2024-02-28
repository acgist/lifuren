/**
 * 字符串
 * 
 * @author acgist
 */
#pragma once

#include <string>
#include <algorithm>

namespace lifuren {
namespace strings {

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

}
}
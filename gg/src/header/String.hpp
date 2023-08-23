#pragma once

#include <string>

namespace lifuren {

/**
 * 字符串格式化
 * 
 * @param format 模板
 * @param flag   占位标记
 * @param args   参数
 * @param length 参数长度
 */
extern void format(std::string& format, std::string& flag, const std::string* args, int length);

}
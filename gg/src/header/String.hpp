#pragma once

#include <string>

namespace lifuren {

namespace gg {

    /**
     * 占位符
     */
    const static std::string FLAG        = "{}";
    /**
     * 占位符的长度
     */
    const static int         FLAG_LENGTH = 2;

    /**
     * 字符串格式化
     * 
     * @param format 模板
     * @param args   参数
     * @param length 参数长度
     */
    extern void format(std::string& format, const std::string* args, int length);

}

}
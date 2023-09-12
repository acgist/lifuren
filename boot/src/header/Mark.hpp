#pragma once

#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace lifuren {

/**
 * 标记
 */
class Mark {

public:
    // 本地文件
    std::string file;
    // 文件散列
    std::string hash;
    // 标签数组
    std::vector<std::string> labels;
public:
    virtual ~Mark() {
    }

public:
    /**
     * @return JSON
     */
    virtual std::string toJSON();

};

/**
 * 文件标记
 */
class MarkFile : public Mark {

};

/**
 * 文本标记
 */
class MarkText : public Mark {

public:
    // 文本名称
    std::string name;
    // 文本内容
    std::string text;

public:
    /**
     * @return JSON
     */
    std::string toJSON() override;

};

}
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
    // JSON序列化
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(lifuren::Mark, file, hash, labels);
public:
    Mark();
    virtual ~Mark();
    /**
     * @return JSON
     */
    virtual std::string toJSON();

};

/**
 * 文件标记
 */
class MarkFile : public Mark {

public:
    MarkFile();
    virtual ~MarkFile();

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
    // JSON序列化
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(lifuren::MarkText, name, text, lifuren::Mark::file, lifuren::Mark::hash, lifuren::Mark::labels);
public:
    MarkText();
    virtual ~MarkText();
    /**
     * @return JSON
     */
    std::string toJSON() override;

};

}
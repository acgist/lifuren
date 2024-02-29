/**
 * 标记
 * 训练文件标签信息
 * 
 * @author acgist
 */
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
    // 标签数组
    std::vector<std::string> labels;
    // JSON序列化
    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(Mark, labels);

public:
    Mark();
    virtual ~Mark();
    /**
     * @param json JSON
     */
    Mark(const std::string& json);

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

public:
    // 本地文件
    std::string file;
        // JSON序列化
    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(MarkFile, file, labels);

public:
    MarkFile();
    virtual ~MarkFile();
    /**
     * @param json JSON
     */
    MarkFile(const std::string& json);

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
    // 标记名称
    std::string label;
    // JSON序列化
    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(MarkText, name, text, label, labels);
    
public:
    MarkText();
    virtual ~MarkText();
    /**
     * @param json JSON
     */
    MarkText(const std::string& json);

public:
    /**
     * @return JSON
     */
    virtual std::string toJSON() override;

};

}

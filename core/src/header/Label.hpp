/**
 * 标签
 * 描述训练数据
 * 
 * @author acgist
 */
#pragma once

#include <map>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace lifuren {

/**
 * 标签
 */
class Label {

public:
    // 标签名称
    std::string name;
    // JSON序列化
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Label, name);

public:
    Label();
    virtual ~Label();
    /**
     * @param json JSON
     */
    Label(const std::string& json);

public:
    /**
     * @return JSON
     */
    virtual std::string toJSON();

};

/**
 * 配置标签
 */
class LabelConfig : public Label {

public:
    // 标签数组
    std::vector<std::string> labels;
    // 下级标签
    std::map<std::string, LabelConfig> children;
    // JSON序列化
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(LabelConfig, name, labels, children);

public:
    LabelConfig();
    virtual ~LabelConfig();
    /**
     * @param json JSON
     */
    LabelConfig(const std::string& json);

public:
    /**
     * @return JSON
     */
    virtual std::string toJSON() override;

};

/**
 * 分词标签
 */
class LabelSegment : public Label {

public:
    // 字数
    int fontSize;
    // 段数
    int segmentSize;
    // 分词规则
    std::vector<int> segmentRule;
    // JSON序列化
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(LabelSegment, name, fontSize, segmentSize, segmentRule);

public:
    LabelSegment();
    virtual ~LabelSegment();
    /**
     * @param json JSON
     */
    LabelSegment(const std::string& json);

public:
    /**
     * @return JSON
     */
    virtual std::string toJSON() override;

};

}

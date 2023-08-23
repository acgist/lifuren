#pragma once

#include <map>
#include <string>
#include <vector>

namespace lifuren {

/**
 * 标签
 */
class Label {

public:
    // 标签名称
    std::string name;
public:
    virtual ~Label() {
    }

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

};

}
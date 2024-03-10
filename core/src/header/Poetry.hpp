/**
 * 诗词
 * 
 * @author acgist
 */
#pragma once

#include <vector>
#include <string>

#include "Label.hpp"

namespace lifuren {

namespace poetry {

/**
 * 段落分隔符
 */
const std::vector<std::string> POETRY_SEGMENT_DELIM = { "、", "，", "。", "？", "！", "；" };
/**
 * 段落美化分隔符
 */
const std::vector<std::string> POETRY_BEAUTIFY_DELIM = { "。", "？", "！", "；" };

/**
 * @param segment 段落
 * 
 * @return 美化后的段落
 */
std::string beautify(const std::string& segment);

}

/**
 * 诗词
 */
class Poetry {

public:
    // 标题
    std::string title;
    // 格律
    std::string rhythmic;
    // 作者
    std::string author;
    // 原始段落
    std::string segment;
    // 朴素段落
    std::string simpleSegment;
    // 分词段落
    std::string participleSegment;
    // 原始段落
    std::vector<std::string> paragraphs;
    // 朴素段落
    std::vector<std::string> simpleParagraphs;
    // 分词段落
    std::vector<std::string> participleParagraphs;
    // 规则：不要释放资源
    LabelText* label = nullptr;
    // JSON序列化
    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(Poetry, title, rhythmic, author, paragraphs);

public:
    /**
     * 预处理
     * 
     * @return *this
     */
    Poetry& preproccess();
    /**
     * 匹配规则
     * 
     * @see #simpleParagraphs
     * 
     * @return 是否匹配成功
     */
    bool matchLabel();
    /**
     * 段落分词
     * 
     * @return 是否分词成功
     */
    bool participle();

};

}
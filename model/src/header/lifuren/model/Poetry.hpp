/**
 * 诗词
 * 
 * TODO: 诗词模型
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_MODEL_POETRY_HPP
#define LFR_HEADER_MODEL_POETRY_HPP

#include <vector>
#include <string>

#include "nlohmann/json.hpp"

#include "lifuren/config/Label.hpp"

namespace lifuren {
namespace poetry  {

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

} // END OF poetry

/**
 * 诗词
 */
class Poetry {

public:
    // 标题
    std::string title;
    // 作者
    std::string author;
    // 格律
    std::string rhythmic;
    // 原始段落
    std::string segment;
    // 朴素段落：没有符号
    std::string simpleSegment;
    // 分词段落：分词分割
    std::string participleSegment;
    // 原始段落
    std::vector<std::string> paragraphs;
    // 朴素段落
    std::vector<std::string> simpleParagraphs;
    // 分词段落
    std::vector<std::string> participleParagraphs;
    // 规则：不要释放资源（全局资源）
    LabelText* label = nullptr;
    // JSON解析
    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(Poetry, title, author, rhythmic, segment, simpleSegment, participleSegment, paragraphs, simpleParagraphs, participleParagraphs);

public:
    /**
     * 预处理
     * 
     * @return 诗词
     */
    Poetry& preproccess();
    /**
     * 匹配规则
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
    /**
     * @param poetry 其他诗词
     * 
     * @return 是否相等
     */
    bool operator==(const Poetry& poetry) const;

};

}

#endif // LFR_HEADER_MODEL_POETRY_HPP

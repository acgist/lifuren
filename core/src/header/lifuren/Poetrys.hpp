/**
 * 诗词工具
 */
#ifndef LFR_HEADER_NLP_POETRYS_HPP
#define LFR_HEADER_NLP_POETRYS_HPP

#include <string>
#include <vector>

#include "nlohmann/json.hpp"

#include "lifuren/Config.hpp"

namespace lifuren {
namespace poetrys {

/**
 * 符号
 */
const std::vector<std::string> POETRY_SYMBOL_DELIM = { "\r", "\n", "\t", " ", "?", ".", "、", "；", "，", "。", "？", "！" };
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
extern std::string beautify(const std::string& segment);

extern std::vector<std::string> toChars(const std::string& poetry);

extern std::vector<std::string> toWords(const std::string& poetry);

extern std::vector<std::string> toSegments(const std::string& poetry);

extern std::string replaceSymbol(const std::string& poetry);

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
    std::string rhythm;
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
    lifuren::config::Rhythm* rhythmPtr = nullptr;
    // JSON解析
    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(Poetry, title, author, rhythm, segment, simpleSegment, participleSegment, paragraphs, simpleParagraphs, participleParagraphs);

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
    bool matchRhythm();
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

} // END OF poetrys
} // END OF lifuren

#endif // LFR_HEADER_NLP_POETRYS_HPP
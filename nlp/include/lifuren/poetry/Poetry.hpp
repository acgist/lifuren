/**
 * 诗词工具
 */
#ifndef LFR_HEADER_NLP_POETRY_HPP
#define LFR_HEADER_NLP_POETRY_HPP

#include <string>
#include <vector>
#include <fstream>

#include "nlohmann/json.hpp"

#include "lifuren/Config.hpp"
#include "lifuren/Thread.hpp"

namespace lifuren {

class EmbeddingClient;

namespace poetry {

// 诗词符号
const std::vector<std::string> POETRY_SIMPLE  = { "、", "，", "。", "？", "！", "；", "：" };

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
    // 朴素段落：没有符号
    std::string simpleSegment;
    // 分词段落
    std::string participleSegment;
    // 原始段落
    std::vector<std::string> paragraphs;
    // 朴素段落
    std::vector<std::string> simpleParagraphs;
    // 分词段落
    std::vector<std::string> participleParagraphs;
    // 格律指针：不要释放（全局资源）
    const lifuren::config::Rhythm* rhythmPtr = nullptr;
    // JSON解析
    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(Poetry, title, author, rhythmic, segment, simpleSegment, participleSegment, paragraphs, simpleParagraphs, participleParagraphs);

public:
    /**
     * 预处理
     * 1. 去掉符号
     * 2. 拼接诗词
     * 
     * @return *this
     */
    Poetry& preproccess();
    /**
     * 匹配格律
     * 
     * @return 是否匹配成功
     */
    bool matchRhythm();
    /**
     * 段落分词
     * 按照格律进行诗句分词
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

extern bool embedding(const std::string& path, std::ofstream& stream, lifuren::thread::ThreadPool& pool);

} // END OF poetry
} // END OF lifuren

#endif // LFR_HEADER_NLP_POETRY_HPP

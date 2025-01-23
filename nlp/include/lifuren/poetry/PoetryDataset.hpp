/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * 诗词数据集
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_NLP_POETRY_DATASET_HPP
#define LFR_HEADER_NLP_POETRY_DATASET_HPP

#include <string>
#include <vector>
#include <fstream>

#include "nlohmann/json.hpp"

#include "lifuren/Thread.hpp"
#include "lifuren/Config.hpp"
#include "lifuren/Dataset.hpp"

namespace lifuren {

class RAGClient;

} // END OF lifuren

namespace lifuren::poetry {

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

extern bool ragEmbedding(const std::shared_ptr<lifuren::RAGClient> ragClient, const std::string& path, const std::string& dataset, std::ofstream& stream, lifuren::thread::ThreadPool& pool);

extern bool pepperEmbedding(const std::string& path, const std::string& dataset, std::ofstream& stream, lifuren::thread::ThreadPool& pool);

/**
 * @param stream 文件流
 * @param vector 诗词数据
 * 
 * @return 是否读完
 */
extern bool read(std::ifstream& stream, std::vector<std::vector<float>>& vector);

/**
 * @param stream 文件流
 * @param vector 诗词数据
 */
extern void write(std::ofstream& stream, const std::vector<std::vector<float>>& vector);

/**
 * @param stream 文件流
 * @param flag   结束标记
 */
inline void writeEnd(std::ofstream& stream, const short& flag) {
    stream.write(reinterpret_cast<const char*>(&flag), sizeof(flag));
}

/**
 * @param dims   向量维度
 * @param vector 向量数据
 * @param rhythm 诗词格律
 * 
 * @return 是否成功
 */
extern bool fillRhythm(const int& dims, std::vector<std::vector<float>>& vector, const lifuren::config::Rhythm* rhythm);

inline torch::Tensor cat(
    std::vector<torch::Tensor>::iterator segmentRule,
    std::vector<torch::Tensor>::iterator participleRule,
    std::vector<torch::Tensor>::iterator beg,
    int index,
    const torch::DeviceType& device
) {
    const int sequenceLength = lifuren::config::CONFIG.poetry.length;
    std::vector<float> indexVector(beg->sizes()[0], 0.0F);
    std::fill(indexVector.begin() + index, indexVector.begin() + index + sequenceLength, 1.0F);
    std::vector<torch::Tensor> sequence;
    sequence.push_back(*segmentRule);
    sequence.push_back(*participleRule);
    sequence.push_back(torch::from_blob(indexVector.data(), beg->sizes(), torch::kFloat32).to(device).clone());
    for(int index = 0; index < sequenceLength; ++index) {
        sequence.push_back(*(beg + index));
    }
    return torch::stack(sequence);
}

/**
 * @param batch_size 批量大小
 * @param path       文件目录
 * 
 * @return 诗词数据集
 */
extern lifuren::dataset::FileDatasetLoader loadFileDatasetLoader(
    const size_t& batch_size,
    const std::string& path
);

} // END OF lifuren::dataset

#endif // END OF LFR_HEADER_NLP_POETRY_DATASET_HPP

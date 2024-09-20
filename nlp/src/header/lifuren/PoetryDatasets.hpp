/**
 * 诗词数据集
 * 
 * @author acgist
 * 
 * TODO:
 * 1. 词嵌入分离直接保存嵌入之后的文件不用每次都要嵌入
 */
#ifndef LFR_HEADER_NLP_POETRY_DATASETS_HPP
#define LFR_HEADER_NLP_POETRY_DATASETS_HPP

#include "lifuren/Poetrys.hpp"
#include "lifuren/Datasets.hpp"

#include <fstream>

namespace lifuren::datasets {

class EmbeddingClient;

namespace poetry {

const short END_OF_POETRY   = -1; // 诗词结束
const short END_OF_DATASETS = -2; // 数据集结束

extern bool read (std::ifstream& stream, std::vector<std::vector<float>>& vector);
extern void write(std::ofstream& stream, std::vector<std::vector<float>>& vector);

}

/**
 * @param batchSize 批量大小
 * @param path      文件目录
 * @param length    向量大小
 * 
 * @return 诗词数据集
 */
inline auto loadPoetryFileDataset(
    const size_t& batchSize,
    const std::string& path,
    const lifuren::EmbeddingClient* client
) -> decltype(auto) {
    return lifuren::datasets::FileDataset(
        batchSize,
        path,
        { ".json" },
        [&client](const std::string& file, std::vector<std::vector<float>>& features) {
            // TODO: 加载embedding.model
        }
    );
}

using PoetryFileDatasetLoader = std::invoke_result<
    decltype(&lifuren::datasets::loadPoetryFileDataset),
    const size_t&,
    const std::string&,
    lifuren::EmbeddingClient*
>::type;

} // END OF lifuren

#endif // END OF LFR_HEADER_NLP_POETRY_DATASETS_HPP

/**
 * 诗词数据集
 * 
 * @author acgist
 * 
 * TODO:
 * 1. 词嵌入分离直接保存嵌入之后的文件不用每次都要嵌入
 * 2. 诗词断句
 * 3. 诗词长度
 */
#ifndef LFR_HEADER_NLP_POETRY_DATASET_HPP
#define LFR_HEADER_NLP_POETRY_DATASET_HPP

#include "lifuren/Poetry.hpp"
#include "lifuren/Dataset.hpp"

#include <fstream>

namespace lifuren::dataset {

class EmbeddingClient;

namespace poetry {

const short END_OF_POETRY  = -1; // 诗词结束
const short END_OF_DATASET = -2; // 数据集结束

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

}

/**
 * @param batch_size 批量大小
 * @param path       文件目录
 * @param client     词嵌入终端
 * 
 * @return 诗词数据集
 */
inline auto loadPoetryFileDataset(
    const size_t& batch_size,
    const std::string& path,
    const lifuren::EmbeddingClient* client
) -> decltype(auto) {
    auto dataset = lifuren::dataset::FileDataset(
        path,
        { ".model" },
        [&client](const std::ifstream& stream, std::vector<torch::Tensor>& labels, std::vector<torch::Tensor>& features) {
            // TODO: 加载embedding.model
        }
    ).map(torch::data::transforms::Stack<>());
    return torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(dataset), batch_size);
}

using PoetryFileDatasetLoader = std::invoke_result<
    decltype(&lifuren::dataset::loadPoetryFileDataset),
    const size_t&,
    const std::string&,
    lifuren::EmbeddingClient*
>::type;

} // END OF lifuren

#endif // END OF LFR_HEADER_NLP_POETRY_DATASET_HPP

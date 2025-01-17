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

#include "lifuren/Dataset.hpp"
#include "lifuren/poetry/Poetry.hpp"

#include <fstream>

#include "spdlog/spdlog.h"

namespace lifuren::dataset {

namespace poetry {

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

} // END OF poetry

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
inline auto loadPoetryFileGANDataset(
    const size_t& batch_size,
    const std::string& path
) -> decltype(auto) {
    auto dataset = lifuren::dataset::FileDataset(
        path,
        [](const std::string& file, std::vector<torch::Tensor>& labels, std::vector<torch::Tensor>& features, const torch::DeviceType& device) {
            std::ifstream stream;
            stream.open(file, std::ios_base::in | std::ios_base::binary);
            if(!stream.is_open()) {
                SPDLOG_WARN("文件打开失败：{}", file);
                stream.close();
                return;
            }
            std::vector<torch::Tensor> data;
            std::vector<std::vector<float>> vector;
            while(!lifuren::dataset::poetry::read(stream, vector)) {
                for(auto& v : vector) {
                    data.push_back(torch::from_blob(v.data(), { static_cast<int>(v.size()) }, torch::kFloat32).to(device).clone());
                }
                int index = 0;
                auto beg = data.begin();
                auto end = data.end();
                auto segmentRule    = beg++;
                auto participleRule = beg++;
                const int sequenceLength = lifuren::config::CONFIG.poetry.length;
                for(; beg + sequenceLength != end; ++beg, ++index) {
                    labels.push_back(*(beg + sequenceLength));
                    features.push_back(cat(segmentRule, participleRule, beg, index, device));
                }
                // EOF
                labels.push_back(torch::zeros({ beg->sizes()[0] }).to(device).clone());
                features.push_back(cat(segmentRule, participleRule, beg, index, device));
                data.clear();
                vector.clear();
            }
            stream.close();
        }
    ).map(torch::data::transforms::Stack<>());
    return torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(dataset), batch_size);
}

using PoetryFileGANDatasetLoader = std::invoke_result<
    decltype(&lifuren::dataset::loadPoetryFileGANDataset),
    const size_t&,
    const std::string&
>::type;

} // END OF lifuren::dataset

#endif // END OF LFR_HEADER_NLP_POETRY_DATASET_HPP

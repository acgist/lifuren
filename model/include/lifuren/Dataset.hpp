/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * 数据集
 * 
 * 数据集已经自动洗牌
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_MODEL_DATASET_HPP
#define LFR_HEADER_MODEL_DATASET_HPP

#include <map>
#include <string>
#include <vector>
#include <fstream>
#include <functional>

#include "torch/data.h"

#include "lifuren/Thread.hpp"

namespace lifuren::dataset {

/**
 * path => [ path/train, path/val, path/test ]
 * 
 * @param path 目录
 * 
 * @return 训练集、验证集、测试集
 */
extern std::vector<std::string> allDataset(const std::string& path);

/**
 * 数据集前置处理
 */
extern bool allDatasetPreprocessing(
    const std::string& path,
    const std::string& model_name,
    std::function<bool(const std::string&, const std::string&, std::ofstream&, lifuren::thread::ThreadPool&)> preprocessing,
    bool model_base = false
);

/**
 * 裸数据集
 */
class RawDataset : public torch::data::Dataset<RawDataset> {

private:
    // 计算设备
    torch::DeviceType device{ torch::DeviceType::CPU };
    // 标签
    std::vector<torch::Tensor> labels;
    // 特征
    std::vector<torch::Tensor> features;

public:
    /**
     * @param labels   标签
     * @param features 特征
     */
    RawDataset(std::vector<torch::Tensor>& labels, std::vector<torch::Tensor>& features);
    virtual ~RawDataset();

public:
    /**
     * @return 数据集大小
     */
    torch::optional<size_t> size() const override;
    /**
     * @param index 索引
     * 
     * @return 数据
     */
    torch::data::Example<> get(size_t index) override;

};

/**
 * 文件数据集
 */
class FileDataset : public torch::data::Dataset<FileDataset> {

private:
    // 计算设备
    torch::DeviceType device{ torch::DeviceType::CPU };
    // 标签
    std::vector<torch::Tensor> labels;
    // 特征
    std::vector<torch::Tensor> features;

public:
    FileDataset() = default;
    FileDataset(const FileDataset& ) = default;
    FileDataset(      FileDataset&&) = default;
    FileDataset& operator=(const FileDataset& ) = delete;
    FileDataset& operator=(      FileDataset&&) = delete;
    /**
     * path/classify1/file1.ext
     * path/classify1/file2.ext
     * path/classify2/file1.ext
     * path/classify2/file2.ext
     * 
     * @param path      数据路径
     * @param suffix    文件后缀
     * @param classify  标签映射
     * @param transform 文件转换
     */
    FileDataset(
        const std::string                 & path,
        const std::vector<std::string>    & suffix,
        const std::map<std::string, float>& classify,
        const std::function<torch::Tensor(const std::string&, const torch::DeviceType&)> transform
    );
    /**
     * path/file.ext
     * 
     * @param path      文件路径
     * @param transform 文件转换
     */
    FileDataset(
        const std::string& path,
        const std::function<void(const std::string&, std::vector<torch::Tensor>&, std::vector<torch::Tensor>&, const torch::DeviceType&)> transform
    );
    /**
     * path/file1.ext
     * path/file2.ext
     * 
     * @param path      数据路径
     * @param suffix    文件后缀
     * @param transform 文件转换
     */
    FileDataset(
        const std::string             & path,
        const std::vector<std::string>& suffix,
        const std::function<void(const std::string&, std::vector<torch::Tensor>&, std::vector<torch::Tensor>&, const torch::DeviceType&)> transform
    );
    virtual ~FileDataset();

public:
    /**
     * @return 数据集大小
     */
    torch::optional<size_t> size() const override;
    /**
     * @param index 索引
     * 
     * @return 数据
     */
    torch::data::Example<> get(size_t index) override;

};

inline auto loadRawDataset(
    const size_t& batch_size,
    std::vector<torch::Tensor>& labels,
    std::vector<torch::Tensor>& features
) -> decltype(auto) {
    auto dataset = lifuren::dataset::RawDataset(labels, features).map(torch::data::transforms::Stack<>());
    return torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(dataset), batch_size);
}

using RawDatasetLoader = std::invoke_result<
    decltype(&lifuren::dataset::loadRawDataset),
    const size_t&,
    std::vector<torch::Tensor>&,
    std::vector<torch::Tensor>&
>::type;

using FileDatasetLoader = decltype(torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
    lifuren::dataset::FileDataset{}.map(torch::data::transforms::Stack<>()),
    torch::data::DataLoaderOptions{}
));

} // END OF lifuren::dataset

#endif // END OF LFR_HEADER_MODEL_DATASET_HPP

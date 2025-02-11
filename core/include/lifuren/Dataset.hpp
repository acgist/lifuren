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
#ifndef LFR_HEADER_CORE_DATASET_HPP
#define LFR_HEADER_CORE_DATASET_HPP

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
 * @return 训练集、验证集、测试集
 */
extern std::vector<std::string> allDataset(
    const std::string& path // 目录
);

/**
 * 数据集预处理
 * 
 * 数据集目录：/data/dataset
 * 数据集实际目录：/data/dataset/train /data/dataset/val /data/dataset/test
 */
extern bool allDatasetPreprocessing(
    const std::string& path,       // 数据集目录
    const std::string& model_name, // 输出文件名称
    std::function<bool(
        const std::string&, // 数据集目录
        const std::string&, // 数据集实际目录
        std::ofstream    &, // 输出文件流
        lifuren::thread::ThreadPool& // 线程池
    )> preprocessing, // 预处理
    bool model_base = false // true-在数据集目录生成文件；false-在数据集实际目录生成文件；
);

/**
 * 裸数据集
 */
class RawDataset : public torch::data::Dataset<RawDataset> {

private:
    torch::DeviceType device{ torch::DeviceType::CPU }; // 计算设备
    std::vector<torch::Tensor> labels;   // 标签
    std::vector<torch::Tensor> features; // 特征

public:
    RawDataset(
        std::vector<torch::Tensor>& labels,  // 标签
        std::vector<torch::Tensor>& features // 特征
    );
    virtual ~RawDataset();

public:
    torch::optional<size_t> size() const override;
    torch::data::Example<> get(size_t index) override;

};

/**
 * 文件数据集
 */
class FileDataset : public torch::data::Dataset<FileDataset> {

private:
    torch::DeviceType device{ torch::DeviceType::CPU }; // 计算设备
    std::vector<torch::Tensor> labels;   // 标签
    std::vector<torch::Tensor> features; // 特征

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
     */
    FileDataset(
        const std::string                 & path,     // 数据路径
        const std::vector<std::string>    & suffix,   // 文件后缀
        const std::map<std::string, float>& classify, // 标签映射
        const std::function<torch::Tensor(
            const std::string      &, // 文件路径
            const torch::DeviceType&  // 计算设备
        )> transform // 文件转换
    );
    /**
     * path/file.ext
     */
    FileDataset(
        const std::string& path, // 文件路径
        const std::function<void(
            const std::string         &, // 文件路径
            std::vector<torch::Tensor>&, // 标签
            std::vector<torch::Tensor>&, // 特征
            const torch::DeviceType   &  // 计算设备
        )> transform // 文件转换
    );
    /**
     * path/file1.ext
     * path/file2.ext
     */
    FileDataset(
        const std::string             & path,   // 数据目录
        const std::vector<std::string>& suffix, // 文件后缀
        const std::function<void(
            const std::string         &, // 文件路径
            std::vector<torch::Tensor>&, // 标签
            std::vector<torch::Tensor>&, // 特征
            const torch::DeviceType   &  // 计算设备
        )> transform // 文件转换
    );
    virtual ~FileDataset();

public:
    torch::optional<size_t> size() const override;
    torch::data::Example<> get(size_t index) override;

};

inline auto loadRawDataset(
    const size_t& batch_size, // 批量大小
    std::vector<torch::Tensor>& labels,  // 标签
    std::vector<torch::Tensor>& features // 特征
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

#endif // END OF LFR_HEADER_CORE_DATASET_HPP

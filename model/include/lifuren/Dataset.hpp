/**
 * Dataset
 * 
 * GANDatasetLoader      : 生成模型
 * StyleDatasetLoader    : 风格迁移模型
 * ClassifyDatasetLoader : 分类模型
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_MODEL_DATASET_HPP
#define LFR_HEADER_MODEL_DATASET_HPP

#include <map>
#include <string>
#include <vector>
#include <functional>

#include "torch/data.h"

namespace lifuren {
namespace dataset {

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
extern bool allDatasetPreprocessing(const std::string& path, std::function<bool(const std::string&)> preprocessing);

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
    /**
     * path/classify1/file1.ext
     * path/classify1/file2.ext
     * path/classify2/file1.ext
     * path/classify2/file2.ext
     * 
     * @param path      数据路径
     * @param exts      文件后缀
     * @param classify  标签映射
     * @param transform 文件转换
     */
    FileDataset(
        const std::string& path,
        const std::vector<std::string>& exts,
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
     * @param exts      文件后缀
     * @param transform 文件转换
     */
    FileDataset(
        const std::string& path,
        const std::vector<std::string>& exts,
        const std::function<void(const std::string&, std::vector<torch::Tensor>&, std::vector<torch::Tensor>&, const torch::DeviceType&)> transform
    );
    /**
     * path/file1.ext
     * path/file1.label
     * path/file2.ext
     * path/file2.label
     * 
     * @param path      数据路径
     * @param label     标记后缀
     * @param exts      文件后缀
     * @param transform 文件转换
     */
    FileDataset(
        const std::string& path,
        const std::string& label,
        const std::vector<std::string>& exts,
        const std::function<void(const std::string&, const std::string&, std::vector<torch::Tensor>&, std::vector<torch::Tensor>&, const torch::DeviceType&)> transform
    );
    /**
     * path/file1.source.ext
     * path/file1.target.ext
     * path/file2.source.ext
     * path/file2.target.ext
     * 
     * @param path      数据路径
     * @param source    原始文件标记
     * @param target    目标文件标记
     * @param exts      文件后缀
     * @param transform 文件转换
     */
    FileDataset(
        const std::string& path,
        const std::string& source,
        const std::string& target,
        const std::vector<std::string>& exts,
        const std::function<void(const std::string&, const std::string&, std::vector<torch::Tensor>&, std::vector<torch::Tensor>&, const torch::DeviceType&)> transform
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

} // END OF dataset
} // END OF lifuren

#endif // LFR_HEADER_MODEL_DATASET_HPP

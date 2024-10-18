/**
 * Dataset
 * 
 * TODO:
 * csv
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_MODEL_DATASET_HPP
#define LFR_HEADER_MODEL_DATASET_HPP

#include <map>
#include <string>
#include <vector>
#include <functional>

#include "torch/torch.h"

namespace lifuren {
namespace dataset {

class RawDataset : public torch::data::Dataset<RawDataset> {

private:
    std::vector<float> labels;
    std::vector<std::vector<float>> features;

public:
    RawDataset(const std::vector<float>& labels, const std::vector<std::vector<float>>& features);
    virtual ~RawDataset();

public:
    /**
     * @return 数据集大小
     */
    torch::optional<size_t> size() const override;
    /**
     * @param index 文件索引
     * 
     * @return Tensor
     */
    torch::data::Example<> get(size_t index) override;

};

class FileDataset : public torch::data::Dataset<FileDataset> {

private:
    // 文件标签
    std::vector<torch::Tensor> labels;
    // 文件路径
    std::vector<std::string> features;
    // 文件转换
    std::function<torch::Tensor(const std::string&)> transform{ nullptr };

public:
    /**
     * @param path      数据路径
     * @param exts      文件后缀
     * @param classify  标签映射
     * @param transform 文件转换
     */
    FileDataset(
        const std::string& path,
        const std::vector<std::string>& exts,
        const std::map<std::string, float>& classify,
        const std::function<torch::Tensor(const std::string&)> transform = nullptr
    );
    /**
     * @param path      数据路径
     * @param exts      文件后缀
     * @param mapping   标签映射
     * @param transform 文件转换
     */
    FileDataset(
        const std::string& path,
        const std::vector<std::string>& exts,
        const std::function<torch::Tensor(const std::string&)> mapping,
        const std::function<torch::Tensor(const std::string&)> transform = nullptr
    );
    virtual ~FileDataset();

public:
    /**
     * @return 数据集大小
     */
    torch::optional<size_t> size() const override;
    /**
     * @param index 文件索引
     * 
     * @return Tensor
     */
    torch::data::Example<> get(size_t index) override;

};

inline auto loadRawDataset(
    const size_t& batch_size,
    const std::vector<float>& labels,
    const std::vector<std::vector<float>>& features
) -> decltype(auto) {
    auto dataset = lifuren::dataset::RawDataset(labels, features).map(torch::data::transforms::Stack<>());
    return torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(dataset), batch_size);
}

using RawDatasetLoader = std::invoke_result<
    decltype(&lifuren::dataset::loadRawDataset),
    const size_t&,
    const std::vector<float>&,
    const std::vector<std::vector<float>>&
>::type;

} // END OF dataset
} // END OF lifuren

#endif // LFR_HEADER_MODEL_DATASET_HPP

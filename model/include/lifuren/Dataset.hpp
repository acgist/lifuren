/**
 * Dataset
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_MODEL_DATASET_HPP
#define LFR_HEADER_MODEL_DATASET_HPP

#include <map>
#include <string>
#include <vector>
#include <fstream>
#include <functional>

#include "torch/torch.h"

namespace lifuren {
namespace dataset {

/**
 * CSV数据集
 */
class CsvDataset : public torch::data::Dataset<CsvDataset> {

private:
    // 标签
    std::vector<torch::Tensor> labels;
    // 特征
    std::vector<torch::Tensor> features;

public:
    CsvDataset(
        const std::string& path,
        const size_t& startRow =  1,
        const size_t& startCol =  1,
        const int   & labelCol = -1,
        const std::string& unknow = "NA"
    );
    virtual ~CsvDataset();

public:
    /**
     * @return 数据集大小
     */
    torch::optional<size_t> size() const override;
    /**
     * @param index 索引
     * 
     * @return Tensor
     */
    torch::data::Example<> get(size_t index) override;
    /**
     * @param index 索引
     */
    torch::Tensor getFeature(size_t index);

public:
    // 重置标记
    static void reset();

};

/**
 * 裸数据集
 */
class RawDataset : public torch::data::Dataset<RawDataset> {

private:
    // 标签
    std::vector<float> labels;
    // 特征
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
     * @param index 索引
     * 
     * @return Tensor
     */
    torch::data::Example<> get(size_t index) override;

};

/**
 * 文件数据集
 */
class FileDataset : public torch::data::Dataset<FileDataset> {

private:
    // 标签
    std::vector<torch::Tensor> labels;
    // 特征
    std::vector<torch::Tensor> features;

public:
    /**
     * @param labels  标签
     * @param features 特征
     */
    FileDataset(
        std::vector<torch::Tensor>& labels,
        std::vector<torch::Tensor>& features
    );
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
        const std::function<torch::Tensor(const std::string&)> transform
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
        const std::function<void(const std::ifstream&, std::vector<torch::Tensor>&, std::vector<torch::Tensor>&)> transform
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
     * @return Tensor
     */
    torch::data::Example<> get(size_t index) override;

};

inline auto loadCsvDataset(
    const size_t& batch_size,
    const std::string path,
    const size_t& startRow =  1,
    const size_t& startCol =  1,
    const int   & labelCol = -1,
    const std::string& unknow = "NA"
) -> decltype(auto) {
    auto dataset = lifuren::dataset::CsvDataset(path, startRow, startCol, labelCol, unknow).map(torch::data::transforms::Stack<>());
    return torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(dataset), batch_size);
}

using CsvDatasetLoader = std::invoke_result<
    decltype(&lifuren::dataset::loadCsvDataset),
    const size_t&,
    const std::string&,
    const size_t&,
    const size_t&,
    const int   &,
    const std::string&
>::type;

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

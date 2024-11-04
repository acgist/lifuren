/**
 * Dataset
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_MODEL_DATASET_HPP
#define LFR_HEADER_MODEL_DATASET_HPP

#include <map>
#include <limits>
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
    // 计算设备
    torch::DeviceType device{ torch::DeviceType::CPU };
    // 标签
    std::vector<torch::Tensor> labels;
    // 特征
    std::vector<torch::Tensor> features;

public:
    /**
     * @param path     文件路径
     * @param startRow 开始行
     * @param startCol 开始列
     * @param labelCol 结束列
     * @param unknow   未知值
     */
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
     * @return 数据
     */
    torch::data::Example<> get(size_t index) override;

public:
    // 重置标记：类型、枚举
    static void reset();
    /**
     * @param path     文件路径
     * @param labels   标签
     * @param features 特征
     * @param startRow 开始行
     * @param startCol 开始列
     * @param labelCol 结束列
     * @param unknow   未知值
     */
    static void loadCSV(
        const std::string& path,
        std::vector<torch::Tensor>& labels,
        std::vector<torch::Tensor>& features,
        torch::DeviceType device = torch::DeviceType::CPU,
        const size_t& startRow =  1,
        const size_t& startCol =  1,
        const int   & labelCol = -1,
        const std::string& unknow = "NA"
    );
    /**
     * @param path   文件路径
     * @param vector 向量
     */
    inline static void loadCSV(
        const std::string& path,
        std::vector<torch::Tensor>& vector,
        torch::DeviceType device = torch::DeviceType::CPU
    ) {
        loadCSV(path, vector, vector, device, 1, 1, std::numeric_limits<int>::max());
    }

};

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
     * @return 数据
     */
    torch::data::Example<> get(size_t index) override;

};

inline auto loadCsvDataset(
    const size_t& batch_size,
    const std::string& path,
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

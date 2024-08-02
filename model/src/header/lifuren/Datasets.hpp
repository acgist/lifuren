/**
 * 文件数据集
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_MODEL_DATASETS_HPP
#define LFR_HEADER_MODEL_DATASETS_HPP

#include <map>
#include <string>
#include <vector>
#include <functional>

#include "torch/torch.h"

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

namespace lifuren  {
namespace datasets {

/**
 * 数据集类型
 */
enum class Type {

    // 训练集
    TRAIN = 0,
    // 验证集
    VAL   = 1,
    // 测试集
    TEST  = 2,

};

/**
 * 数据集分片
 */
class DatasetSharding {

public:
    // 当前训练次数
    int epoch = 0;
    // 数据集类型
    Type type = Type::TRAIN;
    // 训练集占比
    float trainRatio = 0.8F;
    // 验证集占比
    float valRatio   = 0.1F;
    // 测试集占比
    float testRatio  = 0.1F;

public:
    /**
     * @param totalSize 总数据集大小
     * 
     * @return 分片数据集大小
     */
    size_t getSize(size_t totalSize);
    /**
     * @param totalSize 总数据集大小
     * @param index     相对索引
     * 
     * @return 绝对索引
     */
    size_t getIndex(size_t totalSize, size_t index);

};

inline size_t DatasetSharding::getSize(size_t totalSize) {
    assert(this->trainRatio > this->valRatio);
    assert(this->trainRatio > this->testRatio);
    assert(this->trainRatio + this->valRatio + this->testRatio == 1.0F);
    switch(this->type) {
        case Type::TRAIN:
            return static_cast<size_t>(this->trainRatio * totalSize);
        case Type::VAL:
            return this->valRatio * totalSize;
        case Type::TEST:
            return this->testRatio * totalSize;
    }
    return totalSize;
}

inline size_t DatasetSharding::getIndex(size_t totalSize, size_t index) {
    assert(this->trainRatio > this->valRatio);
    assert(this->trainRatio > this->testRatio);
    assert(this->trainRatio + this->valRatio + this->testRatio == 1.0F);
    // K折交叉验证
    switch(this->type) {
        case Type::TRAIN:
        case Type::VAL  : {
            const int shardingCount = (this->trainRatio + this->valRatio) / this->valRatio;
            const int valIndex = this->epoch % shardingCount;
            if(this->type == Type::TRAIN) {
                if(index < this->valRatio * valIndex * totalSize) {
                    // train val train test
                    return index;
                } else {
                    // val train test
                    return this->valRatio * totalSize + index;
                }
            } else {
                return this->valRatio * valIndex * totalSize + index;
            }
            return index;
        }
        case Type::TEST:
            return (this->trainRatio + this->valRatio) * totalSize + index;
    }
    return index;
}

/**
 * 文件数据集
 * 
 * ./类型1/文件列表
 * ./类型2/文件列表
 * ...
 */
class FileDataset : public torch::data::Dataset<FileDataset> {

private:
    // 文件标签
    std::vector<int> labels;
    // 文件路径
    std::vector<std::string> paths;
    // 文件转换
    std::function<torch::Tensor(const std::string&)> fileTransform = nullptr;

public:
    /**
     * @param path          数据路径
     * @param exts          文件后缀
     * @param mapping       标签映射
     * @param fileTransform 文件转换
     */
    FileDataset(
        const std::string& path,
        const std::vector<std::string>& exts,
        const std::map<std::string, int>& mapping,
        const std::function<torch::Tensor(const std::string&)> fileTransform = nullptr
    );

public:
    torch::optional<size_t> size() const override;
    torch::data::Example<> get(size_t index) override;

};

/**
 * 图片数据集
 * 
 * @param width          图片宽度
 * @param height         图片高度
 * @param batch_size     批次大小
 * @param path           图片路径
 * @param image_type     图片格式
 * @param mapping        标签映射
 * @param imageTransform 图片转换
 */
inline auto loadImageFileDataset(
    const int width,
    const int height,
    const size_t batch_size,
    const std::string& path,
    const std::string& image_type,
    const std::map<std::string, int>& mapping,
    const std::function<void(const cv::Mat&)> imageTransform = nullptr
) -> decltype(auto) {
    auto dataset = lifuren::datasets::FileDataset(path, {
        image_type
    }, mapping, [
        width,
        height,
        imageTransform
    ](const std::string& path) -> torch::Tensor {
        cv::Mat image = cv::imread(path);
        cv::resize(image, image, cv::Size(width, height));
        if(imageTransform != nullptr) {
            imageTransform(image);
        }
        torch::Tensor data_tensor = torch::from_blob(image.data, { image.rows, image.cols, 3 }, torch::kByte).permute({2, 0, 1});
        image.release();
        return data_tensor;
    }).map(torch::data::transforms::Stack<>());
    return torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(dataset), batch_size);
}

using ImageFileDatasetLoader = std::invoke_result<
    decltype(&lifuren::datasets::loadImageFileDataset),
    const int,
    const int,
    const size_t,
    const std::string&,
    const std::string&,
    const std::map<std::string, int>&,
    const std::function<void(const cv::Mat&)>
>::type;

/**
 * Tensor数据集
 */
class TensorDataset : public torch::data::Dataset<TensorDataset> {

public:
    // 特征
    torch::Tensor features;
    // 标签
    torch::Tensor labels;

public:
    TensorDataset(torch::Tensor& features, torch::Tensor& labels);

public:
    torch::optional<size_t> size() const override;
    torch::data::Example<> get(size_t index) override;

};

} // END OF datasets
} // END OF lifuren

#endif // LFR_HEADER_MODEL_DATASETS_HPP

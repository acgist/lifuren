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

#include <assert.h>

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

namespace cv {
    class Mat;
};

struct ggml_tensor;

namespace lifuren  {
namespace datasets {

extern ggml_tensor* readImage(const std::string& path, int width, int height, const std::function<void(const cv::Mat&)> imageTransform = nullptr);

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
 * 数据集
 */
class Dataset {

public:
    virtual size_t size() const = 0;
    virtual ggml_tensor* get(size_t index) = 0;

};

/**
 * 文件数据集
 * 
 * ./类型1/文件列表
 * ./类型2/文件列表
 * ...
 */
class FileDataset : public Dataset {

private:
    // 文件标签
    std::vector<int> labels;
    // 文件路径
    std::vector<std::string> paths;
    // 文件转换
    std::function<ggml_tensor*(const std::string&)> fileTransform = nullptr;

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
        const std::function<ggml_tensor*(const std::string&)> fileTransform = nullptr
    );

public:
    size_t size() const override;
    ggml_tensor* get(size_t index) override;

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
    ](const std::string& path) -> ggml_tensor* {
        return readImage(path, width, height, imageTransform);
    });
    return dataset;
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
class TensorDataset : public Dataset {

public:
    // 特征
    ggml_tensor* features;
    // 标签
    ggml_tensor* labels;

public:
    TensorDataset(ggml_tensor* features, ggml_tensor* labels);

public:
    size_t size() const override;
    ggml_tensor* get(size_t index) override;

};

} // END OF datasets
} // END OF lifuren

#endif // LFR_HEADER_MODEL_DATASETS_HPP

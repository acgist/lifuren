/**
 * 数据集工具
 * 
 * TODO: GPU加速
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_MODEL_DATASETS_HPP
#define LFR_HEADER_MODEL_DATASETS_HPP

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <functional>

namespace cv {
    class Mat;
};

namespace lifuren  {
namespace datasets {

/**
 * 图片读取
 * 
 * @param path      图片路径
 * @param data      图片数据
 * @param length    图片数据长度
 * @param width     目标图片宽度
 * @param height    目标图片高度
 * @param transform 图片变换
 */
extern void readImage(
    const std::string& path,
    float * data,
    size_t& length,
    const int& width  = 0,
    const int& height = 0,
    const std::function<void(const cv::Mat&)> transform = nullptr
);

/**
 * 数据集
 */
class Dataset {

protected:
    // 数据总量
    size_t count;
    // 批次数量
    size_t batchSize;

public:
    /**
     * @param batchSize 批次数量
     */
    Dataset(size_t batchSize);
    /**
     * @param count     数据总量
     * @param batchSize 批次数量
     */
    Dataset(size_t count, size_t batchSize);
    virtual ~Dataset();

public:
    /**
     * @return 数据重量
     */
    virtual const size_t& getCount() const;
    /**
     * @return 批次数量
     */
    virtual const size_t& getBatchSize() const;
    /**
     * @return 批次总数
     */
    virtual size_t getBatchCount() const;
    /**
     * 洗牌
     */
    virtual void shuffle();
    /**
     * 获取批次数据
     * 
     * @param batch    批次
     * @param features 数据
     * @param labels   标签
     * 
     * @return 剩余数据数量
     */
    virtual size_t batchGet(size_t batch, void* features, void* labels) const = 0;

};

/**
 * 共享数据集
 */
class ShardingDataset : public Dataset {

private:
    // 原始数据集
    std::shared_ptr<Dataset> source{ nullptr };
    // 索引映射
    std::map<size_t, size_t> indexMapping{};

public:
    /**
     * @param source       原始数据集
     * @param indexMapping 索引映射
     */
    ShardingDataset(
        std::shared_ptr<Dataset> source,
        std::map<size_t, size_t> indexMapping
    );
    ~ShardingDataset();

public:
    inline virtual size_t batchGet(size_t batch, void* features, void* labels) const override {
       return this->source->batchGet(this->indexMapping.at(batch), features, labels);
    }

public:
    /**
     * @param source    原始数据集
     * @param valIndex  验证数据集索引
     * @param valCount  验证数据集总量
     * @param testIndex 测试数据集索引
     * @param testCount 测试数据集总量
     * 
     * @return [train, val, test]
     */
    static std::tuple<ShardingDataset, ShardingDataset, ShardingDataset> make(
        std::shared_ptr<Dataset> source,
        size_t valIndex  = 0,
        size_t valCount  = 1,
        size_t testIndex = 1,
        size_t testCount = 0
    );

};

using SharingDataset = ShardingDataset;

/**
 * Raw数据集
 */
class RawDataset : public Dataset {

public:
    // 特征
    float* features{ nullptr };
    // 特征长度
    size_t feature_size{ 0 };
    // 标签
    float* labels{ nullptr };
    // 标签长度
    size_t label_size{ 0 };

public:
    RawDataset(size_t count, size_t batchSize, float* features, size_t feature_size, float* labels, size_t label_size);
    ~RawDataset();
    
public:
    virtual size_t batchGet(size_t batch, void* features, void* labels) const override;

};

/**
 * 文件数据集
 */
class FileDataset : public Dataset {

private:
    // 特征
    std::vector<std::vector<float>> features{};
    // 标签
    std::vector<float> labels{};
    // 文件转换
    std::function<void(const std::string&, std::vector<std::vector<float>>&)> transform{ nullptr };

public:
    /**
     * @param path      数据路径
     * @param exts      文件后缀
     * @param transform 文件转换
     * @param mapping   标签映射
     */
    FileDataset(
        const size_t& batchSize,
        const std::string& path,
        const std::vector<std::string>& exts,
        std::function<void(const std::string&, std::vector<std::vector<float>>&)> transform,
        const std::map<std::string, float>& mapping = {}
    );
    ~FileDataset();

public:
    virtual size_t batchGet(size_t batch, void* features, void* labels) const override;

};

/**
 * 图片数据集
 * 
 * @param width      图片宽度
 * @param height     图片高度
 * @param batch_size 批次大小
 * @param path       图片路径
 * @param image_type 图片格式
 * @param mapping    标签映射
 * @param transform  图片转换
 */
inline auto loadImageFileDataset(
    const int& width,
    const int& height,
    const size_t& batch_size,
    const std::string& path,
    const std::string& image_type,
    const std::map<std::string, float>& mapping,
    const std::function<void(const cv::Mat&)> transform = nullptr
) -> decltype(auto) {
    auto dataset = lifuren::datasets::FileDataset(
    batch_size,
    path,
    { image_type },
    [width, height, transform](const std::string& path, std::vector<std::vector<float>>& features) {
        std::vector<float> feature;
        feature.resize(width * height * 3);
        size_t length{ 0 };
        lifuren::datasets::readImage(path, feature.data(), length, width, height, transform);
        features.push_back(std::move(feature));
    },
    mapping);
    return dataset;
}

using ImageFileDatasetLoader = std::invoke_result<
    decltype(&lifuren::datasets::loadImageFileDataset),
    const int&,
    const int&,
    const size_t&,
    const std::string&,
    const std::string&,
    const std::map<std::string, float>&,
    const std::function<void(const cv::Mat&)>
>::type;

} // END OF datasets
} // END OF lifuren

#endif // LFR_HEADER_MODEL_DATASETS_HPP

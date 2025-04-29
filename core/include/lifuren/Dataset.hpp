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
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_CORE_DATASET_HPP
#define LFR_HEADER_CORE_DATASET_HPP

#ifndef LFT_SAMPLER
#define LFT_RND_SAMPLER torch::data::samplers::RandomSampler
#define LFT_SEQ_SAMPLER torch::data::samplers::SequentialSampler
#endif

#include <string>
#include <vector>
#include <functional>

#include "torch/data.h"

// TODO: 补帧
// 图片配置
#ifndef LFR_IMAGE_CONFIG
#define LFR_IMAGE_CONFIG
#define LFR_IMAGE_WIDTH      270 // 宽度：16:9
#define LFR_IMAGE_HEIGHT     480 // 高度：16:9
#define LFR_VIDEO_FPS        24  // 帧率
#define LFR_VIDEO_FRAME_SIZE 120 // 帧数
#define LFR_VIDEO_FRAME_STEP 4   // 帧数间隔
#endif

namespace cv {

    class Mat;

} // END OF cv

namespace lifuren::dataset {

/**
 * /dataset => [ /dataset/train, /dataset/val, /dataset/test ]
 * 
 * @param path 数据集上级目录
 * 
 * @return 训练集、验证集、测试集
 */
extern std::vector<std::string> allDataset(const std::string& path);

/**
 * 数据集
 */
class Dataset : public torch::data::Dataset<Dataset> {

private:
    torch::DeviceType device{ torch::DeviceType::CPU }; // 计算设备
    std::vector<torch::Tensor> labels;   // 标签
    std::vector<torch::Tensor> features; // 特征

public:
    Dataset() = default;
    Dataset(const Dataset& ) = default;
    Dataset(      Dataset&&) = default;
    Dataset& operator=(const Dataset& ) = delete;
    Dataset& operator=(      Dataset&&) = delete;
    /**
     * @param labels   标签
     * @param features 特征
     */
    Dataset(
        std::vector<torch::Tensor>& labels,
        std::vector<torch::Tensor>& features
    );
    /**
     * /path/file1.suffix
     * /path/file2.suffix
     * ...
     * 
     * @param path      数据集目录
     * @param suffix    文件后缀
     * @param transform 文件转换函数：void(文件路径, 标签, 特征, 计算设备)
     * @param complete  完成回调：void(标签, 特征, 计算设备)
     */
    Dataset(
        const std::string             & path,
        const std::vector<std::string>& suffix,
        const std::function<void(const std::string&, std::vector<torch::Tensor>&, std::vector<torch::Tensor>&, const torch::DeviceType&)> transform,
        const std::function<void(std::vector<torch::Tensor>&, std::vector<torch::Tensor>&, const torch::DeviceType&)> complete = nullptr
    );
    virtual ~Dataset();

public:
    torch::optional<size_t> size() const override;
    torch::data::Example<> get(size_t index) override;

};

using RndDatasetLoader = decltype(torch::data::make_data_loader<LFT_RND_SAMPLER>(
    lifuren::dataset::Dataset{}.map(torch::data::transforms::Stack<>()),
    torch::data::DataLoaderOptions{}
));

using SeqDatasetLoader = decltype(torch::data::make_data_loader<LFT_SEQ_SAMPLER>(
    lifuren::dataset::Dataset{}.map(torch::data::transforms::Stack<>()),
    torch::data::DataLoaderOptions{}
));

namespace image {

/**
 * @param image  图片
 * @param width  目标图片宽度
 * @param height 目标图片高度
 */
extern void resize(cv::Mat& image, const int width, const int height);

/**
 * @param image 图片
 * 
 * @return 图片张量
 */
extern torch::Tensor mat_to_tensor(const cv::Mat& image);

/**
 * @param image  图片：需要提前申请空间
 * @param tensor 图片张量
 */
extern void tensor_to_mat(cv::Mat& image, const torch::Tensor& tensor);

/**
 * @param width      目标图片宽度
 * @param height     目标图片高度
 * @param batch_size 批量大小
 * @param path       数据集路径
 * 
 * @return 图片数据集
 */
extern lifuren::dataset::SeqDatasetLoader loadWudaoziDatasetLoader(const int width, const int height, const size_t batch_size, const std::string& path);

} // END OF image

} // END OF lifuren::dataset

#endif // END OF LFR_HEADER_CORE_DATASET_HPP

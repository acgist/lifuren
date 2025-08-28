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

/**
 * 图片配置
 * 
 * 模型大小
 * 
 *              params size * kFloat16 / KB   / MB   / GB   = 0.18GB
 * model size = 100000000   * 2        / 1024 / 1024 / 1024 = 0.18GB
 * 
 *              params size * kFloat32 / KB   / MB   / GB   = 0.37GB
 * model size = 100000000   * 4        / 1024 / 1024 / 1024 = 0.37GB
 * 
 * 一批训练数据大小
 * 
 *                height * width * channel * kFloat16 * batch_frames * batch / KB   / MB   / GB   = 2.20 GB/Batch
 * dataset size = 128    * 256   * 3       * 2        * 120          * 100   / 1024 / 1024 / 1024 = 2.20 GB/Batch
 * 
 *                height * width * channel * kFloat32 * batch_frames * batch / KB   / MB   / GB   = 4.39 GB/Batch
 * dataset size = 128    * 256   * 3       * 4        * 120          * 100   / 1024 / 1024 / 1024 = 4.39 GB/Batch
 * 
 * 一小时数据集大小
 * 
 *                height * width * channel * kFloat16 * fps * second * minute / step / KB   / MB   / GB   = 15.82 GB/Hour
 * dataset size = 128    * 256   * 3       * 2        * 24  * 60     * 60     / 1    / 1024 / 1024 / 1024 = 15.82 GB/Hour
 * 
 *                height * width * channel * kFloat32 * fps * second * minute / step / KB   / MB   / GB   = 49.43 GB/Hour
 * dataset size = 128    * 256   * 3       * 4        * 24  * 60     * 60     / 1    / 1024 / 1024 / 1024 = 31.64 GB/Hour
 */
#ifndef LFR_IMAGE_CONFIG
#define LFR_IMAGE_CONFIG
#define LFR_IMAGE_WIDTH      128 // 宽度
#define LFR_IMAGE_HEIGHT     256 // 高度
#define LFR_VIDEO_FPS         24 // 帧率
#define LFR_VIDEO_DIFF        30 // 上下文切换差异
#define LFR_VIDEO_FRAME        5 // 前置帧数
#define LFR_VIDEO_FRAME_MIN   24 // 最小帧数
#define LFR_VIDEO_FRAME_MAX  240 // 最大帧数
#define LFR_VIDEO_FRAME_STEP   2 // 帧数间隔（抽帧）
#define LFR_VIDEO_BLACK_MEAN  10 // 黑色二值化阈值
#endif

namespace cv {

    class Mat;

} // END OF cv

namespace lifuren {

/**
 * @return 设备类型
 */
extern torch::DeviceType get_device();

namespace dataset {

// void(文件路径, 标签, 特征, 计算设备)
using Transform = std::function<void(const std::string&, std::vector<torch::Tensor>&, std::vector<torch::Tensor>&, const torch::DeviceType&)>;

/**
 * 数据集
 */
class Dataset : public torch::data::Dataset<Dataset> {

private:
    size_t batch_size; // 批次数量
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
     * @param batch_size 批次数量
     * @param labels     标签
     * @param features   特征
     */
    Dataset(
        const size_t batch_size,
        std::vector<torch::Tensor>& labels,
        std::vector<torch::Tensor>& features
    );
    /**
     * /path/file1.suffix
     * /path/file2.suffix
     * ...
     * 
     * @param batch_size 批次数量
     * @param path       数据集目录
     * @param suffix     文件后缀
     * @param transform  文件转换函数
     */
    Dataset(
        const size_t batch_size,
        const std::string& path,
        const std::vector<std::string>& suffix,
        const Transform transform
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
 * @param batch_size 批次数量
 * @param path       数据集路径
 * 
 * TODO: 分批加载
 * 
 * feature = [ prev_frame, next_frame ]
 * label   = [ time ]
 * 
 * @return 图片数据集
 */
extern lifuren::dataset::RndDatasetLoader loadWudaoziDatasetLoader(const int width, const int height, const size_t batch_size, const std::string& path);

} // END OF image

} // END OF dataset

} // END OF lifuren

#endif // END OF LFR_HEADER_CORE_DATASET_HPP

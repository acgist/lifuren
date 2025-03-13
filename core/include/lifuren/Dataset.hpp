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

#ifndef LFR_DATASET_PCM_CONFIG
// PCM分段大小：1 ms mono 16 bit = 48000 * 16 * 1 / 8 / 1000 * 1 = 96 byte = 48 short
#define LFR_DATASET_PCM_LENGTH 480
#define LFR_DATASET_PCM_DIM_1 201
// 480 = 7 | 4800 = 61 | 48000 = 601
#define LFR_DATASET_PCM_DIM_2 7
#define LFR_DATASET_PCM_DIM_3 2
#define LFR_DATASET_PCM_BATCH_SIZE 100
#endif

// 图片配置
#ifndef LFR_IMAGE_CONFIG
#define LFR_IMAGE_CONFIG
#define LFR_IMAGE_WIDTH  640 // 宽度
#define LFR_IMAGE_HEIGHT 480 // 高度
#endif

namespace cv {

    class Mat;

} // END OF cv

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
 * 数据集上级目录：/data/dataset
 * 数据集目录：/data/dataset/train /data/dataset/val /data/dataset/test
 */
extern bool allDatasetPreprocessing(
    const std::string& path,
    std::function<bool(
        const std::string&, // 数据集上级目录
        const std::string&, // 数据集目录
        lifuren::thread::ThreadPool& // 线程池
    )> preprocessing // 预处理
);

/**
 * 数据集预处理
 * 
 * 数据集上级目录：/data/dataset
 * 数据集目录：/data/dataset/train /data/dataset/val /data/dataset/test
 */
extern bool allDatasetPreprocessing(
    const std::string& path,       // 数据集目录
    const std::string& model_name, // 输出文件名称
    std::function<bool(
        const std::string&, // 数据集上级目录
        const std::string&, // 数据集目录
        std::ofstream    &, // 输出文件流
        lifuren::thread::ThreadPool& // 线程池
    )> preprocessing, // 预处理
    bool model_base = false // true-在数据集上级目录生成文件；false-在数据集目录生成文件；
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

namespace audio {

/**
 * 音频文件转为PCM文件
 * 
 * 支持音频文件格式：AAC/MP3/FLAC
 * 
 * PCM文件格式：48000Hz mono 16bit
 * 
 * @return <是否成功, PCM文件路径>
 */
extern std::tuple<bool, std::string> toPcm(
    const std::string& audioFile // 音频文件
);

/**
 * PCM文件转为音频文件
 * 
 * PCM文件格式：48000Hz mono 16bit
 * 
 * @return <是否成功, 音频文件路径>
 */
extern std::tuple<bool, std::string> toFile(
    const std::string& pcmFile // PCM文件
);

/**
 * 短时傅里叶变换
 * 
 * [1, 201, 61, 2[实部, 虚部]]
 * 
 * @return 张量
 */
extern torch::Tensor pcm_stft(
    std::vector<short>& pcm, // PCM数据
    int n_fft    = 400, // 傅里叶变换的大小
    int hop_size = 80,  // 相邻滑动窗口帧之间的距离
    int win_size = 400  // 窗口帧和STFT滤波器的大小
);

/**
 * 短时傅里叶逆变换
 * 
 * @return PCM
 */
extern std::vector<short> pcm_istft(
    const torch::Tensor& tensor, // 张量
    int n_fft    = 400, // 傅里叶变换的大小
    int hop_size = 80,  // 相邻滑动窗口帧之间的距离
    int win_size = 400  // 窗口帧和STFT滤波器的大小
);

/**
 * 音频嵌入
 * 
 * @return 是否成功
 */
extern bool embedding(
    const std::string& path,    // 数据集上级目录
    const std::string& dataset, // 数据集目录
    std::ofstream    & stream,  // 嵌入文件流
    lifuren::thread::ThreadPool& pool // 线程池
);

/**
 * @return 音频数据集
 */
extern lifuren::dataset::FileDatasetLoader loadFileDatasetLoader(
    const size_t batch_size, // 批量大小
    const std::string& path, // 数据集路径
    const int dim_1 = LFR_DATASET_PCM_DIM_1, // 维度1
    const int dim_2 = LFR_DATASET_PCM_DIM_2, // 维度2
    const int dim_3 = LFR_DATASET_PCM_DIM_3  // 维度3
);

} // END OF audio

namespace image {

/**
 * 修改图片大小
 */
extern void resize(
    cv::Mat& image,  // 图片数据
    const int width, // 目标宽度
    const int height // 目标高度
);

/**
 * 图片转为张量
 * 
 * @return 图片张量
 */
extern torch::Tensor feature(
    const cv::Mat& image, // 图片数据
    const int width,      // 图片宽度
    const int height      // 图片高度
);

/**
 * 张量转为图片
 */
extern void tensor_to_mat(
    cv::Mat& mat, // 图片数据：需要提前申请空间
    const torch::Tensor& tensor // 图片张量
);

/**
 * @return 图片数据集
 */
extern lifuren::dataset::FileDatasetLoader loadFileDatasetLoader(
    const int width,  // 图片宽度
    const int height, // 图片高度
    const size_t batch_size, // 批量大小
    const std::string& path // 数据集路径
);

/**
 * @return 图片数据集
 */
extern lifuren::dataset::FileDatasetLoader loadFileDatasetLoader(
    const int width,  // 图片宽度
    const int height, // 图片高度
    const size_t batch_size, // 批量大小
    const std::string& path, // 数据集路径
    const std::map<std::string, float>& classify // 图片分类
);

} // END OF image

} // END OF lifuren::dataset

#endif // END OF LFR_HEADER_CORE_DATASET_HPP

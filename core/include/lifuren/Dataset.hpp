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

#include <map>
#include <string>
#include <vector>
#include <fstream>
#include <functional>

#include "torch/data.h"

#include "lifuren/Thread.hpp"

// 音频配置
// PCM数据大小：1 ms mono 16 bit = 48000 * 16 * 1 / 8 / 1000 * 1 = 96 byte = 48 short
#ifndef LFR_AUDIO_PCM_CONFIG
#define LFR_AUDIO_PCM_CONFIG
#define LFR_AUDIO_PCM_LENGTH 480
#endif

// 图片配置
#ifndef LFR_IMAGE_CONFIG
#define LFR_IMAGE_CONFIG
#define LFR_IMAGE_WIDTH  640
#define LFR_IMAGE_HEIGHT 480
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
 * 数据集预处理
 * 
 * @param path       数据集上级目录：/dataset
 * @param model_name 输出文件名称
 * @param preprocess 预处理函数：是否成功(数据集上级目录, 数据集目录, 输出文件流, 线程池)
 * @param model_base 是否在数据集上级目录生成文件
 *  
 * @return 是否成功
 */
extern bool allDatasetPreprocess(
    const std::string& path,
    const std::string& model_name,
    std::function<bool(const std::string&, const std::string&, std::ofstream&, lifuren::thread::ThreadPool&)> preprocess,
    bool model_base = false
);

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
    /**
     * /path/file1.l_suffix
     * /path/file1.f_suffix
     * /path/file2.l_suffix
     * /path/file2.f_suffix
     * ...
     * 
     * @param path      数据集目录
     * @param l_suffix  标签文件后缀
     * @param f_suffix  特征文件后缀
     * @param transform 文件转换函数：void(标签文件路径, 特征文件路径, 标签, 特征, 计算设备)
     * @param complete  完成回调：void(标签, 特征, 计算设备)
     */
    Dataset(
        const std::string             & path,
        const std::string             & l_suffix,
        const std::vector<std::string>& f_suffix,
        const std::function<void(const std::string&, const std::string&, std::vector<torch::Tensor>&, std::vector<torch::Tensor>&, const torch::DeviceType&)> transform,
        const std::function<void(std::vector<torch::Tensor>&, std::vector<torch::Tensor>&, const torch::DeviceType&)> complete = nullptr
    );
    /**
     * /path/classify1/file1.suffix
     * /path/classify1/file2.suffix
     * /path/classify2/file1.suffix
     * /path/classify2/file2.suffix
     * ...
     * 
     * @param path      数据集目录
     * @param suffix    文件后缀
     * @param classify  标签映射
     * @param transform 文件转换函数：张量(文件路径, 计算设备)
     * @param complete  完成回调：void(计算设备)
     */
    Dataset(
        const std::string                 & path,
        const std::vector<std::string>    & suffix,
        const std::map<std::string, float>& classify,
        const std::function<torch::Tensor(const std::string&, const torch::DeviceType&)> transform,
        const std::function<void(const torch::DeviceType&)> complete = nullptr
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

namespace audio {

/**
 * 音频文件转为PCM文件
 * 
 * 支持音频文件格式：AAC/MP3/FLAC
 * PCM文件格式：48000Hz mono 16bit
 * 
 * @param audioFile 音频文件路径
 * 
 * @return <是否成功, PCM文件路径>
 */
extern std::tuple<bool, std::string> toPcm(const std::string& audioFile);

/**
 * PCM文件转为音频文件
 * 
 * PCM文件格式：48000Hz mono 16bit
 * 
 * @param pcmFile PCM文件路径
 * 
 * @return <是否成功, 音频文件路径>
 */
extern std::tuple<bool, std::string> toFile(const std::string& pcmFile);

/**
 * 短时傅里叶变换
 * 
 * 201 = win_size / 2 + 1
 * 480 = 7 | 4800 = 61 | 48000 = 601
 * [1, 201, 61, 2[实部, 虚部]]
 * 
 * @param pcm      PCM数据
 * @param n_fft    傅里叶变换的大小
 * @param hop_size 相邻滑动窗口帧之间的距离
 * @param win_size 窗口帧和STFT滤波器的大小
 * 
 * @return 张量
 */
extern torch::Tensor pcm_stft(
    std::vector<short>& pcm,
    int n_fft    = 400,
    int hop_size = 80,
    int win_size = 400
);

/**
 * 短时傅里叶逆变换
 * 
 * @param tensor   张量
 * @param n_fft    傅里叶变换的大小
 * @param hop_size 相邻滑动窗口帧之间的距离
 * @param win_size 窗口帧和STFT滤波器的大小
 * 
 * @return PCM数据
 */
extern std::vector<short> pcm_istft(
    const torch::Tensor& tensor,
    int n_fft    = 400,
    int hop_size = 80,
    int win_size = 400
);

/**
 * 师旷音频嵌入
 * 
 * @param path    数据集上级目录
 * @param dataset 数据集目录
 * @param stream  嵌入文件流
 * @param pool    线程池
 * 
 * @return 是否成功
 */
extern bool embedding_shikuang(const std::string& path, const std::string& dataset, std::ofstream& stream, lifuren::thread::ThreadPool& pool);

/**
 * @param batch_size 批量大小
 * @param path       数据集路径
 * 
 * @return 音频数据集
 */
extern lifuren::dataset::SeqDatasetLoader loadShikuangDatasetLoader(const size_t batch_size, const std::string& path);

} // END OF audio

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
extern lifuren::dataset::RndDatasetLoader loadWudaoziDatasetLoader(const int width, const int height, const size_t batch_size, const std::string& path);

/**
 * @param width      目标图片宽度
 * @param height     目标图片高度
 * @param batch_size 批量大小
 * @param path       数据集路径
 * @param classify   图片分类
 * 
 * @return 图片数据集
 */
extern lifuren::dataset::RndDatasetLoader loadClassifyDatasetLoader(const int width, const int height, const size_t batch_size, const std::string& path, const std::map<std::string, float>& classify);

} // END OF image

} // END OF lifuren::dataset

#endif // END OF LFR_HEADER_CORE_DATASET_HPP

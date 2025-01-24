/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * 音频数据集
 * 
 * https://pytorch.org/docs/stable/generated/torch.stft.html
 * https://pytorch.org/docs/stable/generated/torch.istft.html
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_CV_AUDIO_DATASET_HPP
#define LFR_HEADER_CV_AUDIO_DATASET_HPP

#include "lifuren/Thread.hpp"
#include "lifuren/Dataset.hpp"

// 10 ms = 48000 * 16 * 1 / 8 / 1000 * 10 = 960 byte = 480 short
#ifndef DATASET_PCM_LENGTH
#define DATASET_PCM_LENGTH 480
#endif

namespace lifuren::audio {

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
 * @return [mag, pha, com]
 */
extern std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> pcm_mag_pha_stft(
    std::vector<short>& pcm, // PCM数据
    int n_fft    = 400,      // 傅里叶变换的大小
    int hop_size = 100,      // 相邻滑动窗口帧之间的距离：floor(n_fft / 4)
    int win_size = 400,      // 窗口帧和STFT滤波器的大小：n_fft
    float compress_factor = 1.0 // 压缩因子
);

/**
 * 短时傅里叶逆变换
 * 
 * @return PCM
 */
extern std::vector<short> pcm_mag_pha_istft(
    torch::Tensor mag, // mag
    torch::Tensor pha, // pha
    int n_fft    = 400, // 傅里叶变换的大小
    int hop_size = 100, // 相邻滑动窗口帧之间的距离：floor(n_fft / 4)
    int win_size = 400, // 窗口帧和STFT滤波器的大小：n_fft
    float compress_factor = 1.0 // 压缩因子
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

extern lifuren::dataset::FileDatasetLoader loadFileDatasetLoader(
    const size_t batch_size, // 批量大小
    const std::string& path  // 数据集路径
);

} // END OF lifuren::audio

#endif // END OF LFR_HEADER_CV_AUDIO_DATASET_HPP

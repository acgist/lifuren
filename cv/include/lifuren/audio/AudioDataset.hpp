/**
 * 音频数据集
 */
#ifndef LFR_HEADER_CV_AUDIO_DATASET_HPP
#define LFR_HEADER_CV_AUDIO_DATASET_HPP

#include "lifuren/Thread.hpp"
#include "lifuren/Dataset.hpp"

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
 * 短时傅里叶变换
 * https://pytorch.org/docs/stable/generated/torch.stft.html
 * 
 * @param pcm_norm        一维时间序列或二维时间序列批次
 * @param n_fft           傅里叶变换的大小
 * @param hop_size        相邻滑动窗口帧之间的距离：floor(n_fft / 4)
 * @param win_size        窗口帧和STFT滤波器的大小：n_fft
 * @param compress_factor 压缩因子
 * 
 * 480 mag sizes = pha sizes = [1, 201, 5]
 * 
 * @return [mag, pha, com]
 */
extern std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> mag_pha_stft(
    torch::Tensor pcm_norm,
    int n_fft    = 400,
    int hop_size = 100,
    int win_size = 400,
    float compress_factor = 1.0
);

/**
 * @param pcm         PCM
 * @param norm_factor 归一化因子
 * ...
 * 
 * @return [mag, pha, com]
 */
extern std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> pcm_mag_pha_stft(
    std::vector<short>& pcm,
    float& norm_factor,
    int n_fft    = 400,
    int hop_size = 100,
    int win_size = 400,
    float compress_factor = 1.0
);

/**
 * 短时傅里叶逆变换
 * https://pytorch.org/docs/stable/generated/torch.istft.html
 * 
 * @param mag             mag
 * @param pha             pha
 * @param n_fft           傅里叶变换的大小
 * @param hop_size        相邻滑动窗口帧之间的距离：floor(n_fft / 4)
 * @param win_size        窗口帧和STFT滤波器的大小：n_fft
 * @param compress_factor 压缩因子
 * 
 * @return PCM
 */
extern torch::Tensor mag_pha_istft(
    torch::Tensor mag,
    torch::Tensor pha,
    int n_fft    = 400,
    int hop_size = 100,
    int win_size = 400,
    float compress_factor = 1.0
);

/**
 * @param mag         mag
 * @param pha         pha
 * @param norm_factor 归一化因子
 * ...
 * 
 * @return PCM
 */
extern std::vector<short> pcm_mag_pha_istft(
    torch::Tensor mag,
    torch::Tensor pha,
    const float& norm_factor,
    int n_fft    = 400,
    int hop_size = 100,
    int win_size = 400,
    float compress_factor = 1.0
);

extern lifuren::dataset::FileDatasetLoader loadFileDatasetLoader(
    const size_t& batch_size,
    const std::string& path
);

} // END OF lifuren::audio

#endif // END OF LFR_HEADER_CV_AUDIO_DATASET_HPP

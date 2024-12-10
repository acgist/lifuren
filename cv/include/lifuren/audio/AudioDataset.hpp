/**
 * 音频数据集
 */
#ifndef LFR_HEADER_CV_AUDIO_DATASET_HPP
#define LFR_HEADER_CV_AUDIO_DATASET_HPP

#include "lifuren/Dataset.hpp"

namespace lifuren::dataset {

namespace audio {

extern torch::Tensor feature(const std::string& file, const torch::DeviceType& type);

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
    std::vector<short> pcm,
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

}
    
} // END OF lifuren::dataset

#endif // END OF LFR_HEADER_CV_AUDIO_DATASET_HPP

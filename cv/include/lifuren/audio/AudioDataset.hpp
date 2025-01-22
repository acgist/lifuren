/**
 * 音频数据集
 */
#ifndef LFR_HEADER_CV_AUDIO_DATASET_HPP
#define LFR_HEADER_CV_AUDIO_DATASET_HPP

#include "lifuren/Dataset.hpp"

#ifndef DATASET_PCM_LENGTH
#define DATASET_PCM_LENGTH 480
#endif

namespace lifuren::dataset {

namespace audio {

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

extern torch::Tensor feature(const int& length, const std::string& file, const torch::DeviceType& type);

}

inline FileDatasetLoader loadAudioFileStyleDataset(
    const size_t& batch_size,
    const std::string& path
) {
    auto dataset = lifuren::dataset::FileDataset(
        path,
        [](const std::string& file, std::vector<torch::Tensor>& labels, std::vector<torch::Tensor>& features, const torch::DeviceType& device) {
            std::ifstream stream;
            stream.open(file, std::ios_base::binary);
            if(!stream.is_open()) {
                stream.close();
                return;
            }
            float source_norm_factor;
            float target_norm_factor;
            torch::Tensor source_tensor = torch::zeros({ 1 });
            torch::Tensor target_tensor = torch::zeros({ 1 });
            while(
                stream.read(reinterpret_cast<char*>(&source_norm_factor), sizeof(float)) &&
                stream.read(reinterpret_cast<char*>(source_tensor.data_ptr()), source_tensor.numel() * source_tensor.element_size()) &&
                stream.read(reinterpret_cast<char*>(&target_norm_factor), sizeof(float)) &&
                stream.read(reinterpret_cast<char*>(target_tensor.data_ptr()), target_tensor.numel() * target_tensor.element_size())
            ) {
                features.push_back(std::move(source_tensor.to(device)));
                labels  .push_back(std::move(target_tensor.to(device)));
            }
            stream.close();
        }
    ).map(torch::data::transforms::Stack<>());
    return torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(dataset), batch_size);
}

} // END OF lifuren::dataset

#endif // END OF LFR_HEADER_CV_AUDIO_DATASET_HPP

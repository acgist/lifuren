/**
 * 音频数据集
 */
#ifndef LFR_HEADER_CV_AUDIO_DATASET_HPP
#define LFR_HEADER_CV_AUDIO_DATASET_HPP

#include "lifuren/Dataset.hpp"

#include <fstream>

#include "spdlog/spdlog.h"

#include "nlohmann/json.hpp"

#include "lifuren/File.hpp"
#include "lifuren/String.hpp"

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

inline auto loadAudioFileGANDataset(
    const int& length,
    const size_t& batch_size,
    const std::string& path
) -> decltype(auto) {
    auto dataset = lifuren::dataset::FileDataset(
        path,
        ".json",
        { ".pcm" },
        [length] (const std::string& audio_file, const std::string& label_file, std::vector<torch::Tensor>& labels, std::vector<torch::Tensor>& features, const torch::DeviceType& device) -> void {
            const std::string content = std::move(lifuren::file::loadFile(label_file));
            if(content.empty()) {
                SPDLOG_WARN("音频文件标记无效：{}", label_file);
                return;
            }
            const nlohmann::json prompts = std::move(nlohmann::json::parse(content));
            auto vector = std::move(lifuren::string::embedding(prompts.get<std::vector<std::string>>()));
            if(vector.empty()) {
                SPDLOG_WARN("音频文件标记无效：{}", label_file);
                return;
            }
            labels.push_back(torch::from_blob(vector.data(), { static_cast<int>(vector.size()) }, torch::kFloat32).clone().to(device));
            features.push_back(std::move(lifuren::dataset::audio::feature(length, audio_file, device)));
        }
    ).map(torch::data::transforms::Stack<>());
    return torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(dataset), batch_size);
}

using AudioFileGANDatasetLoader = std::invoke_result<
    decltype(&lifuren::dataset::loadAudioFileGANDataset),
    const int&,
    const size_t&,
    const std::string&
>::type;

inline auto loadAudioFileStyleDataset(
    const size_t& batch_size,
    const std::string& path
) -> decltype(auto) {
    auto dataset = lifuren::dataset::FileDataset(
        path,
        "source",
        "target",
        { ".pcm" },
        [] (const std::string& source, const std::string& target, std::vector<torch::Tensor>& labels, std::vector<torch::Tensor>& features, const torch::DeviceType& device) -> void {
            std::ifstream source_stream;
            std::ifstream target_stream;
            source_stream.open(source, std::ios_base::binary);
            target_stream.open(target, std::ios_base::binary);
            if(!source_stream.is_open() || !target_stream.is_open()) {
                SPDLOG_DEBUG("打开文件失败：{} - {}", source, target);
                source_stream.close();
                target_stream.close();
                return;
            }
            std::vector<short> source_pcm;
            std::vector<short> target_pcm;
            source_pcm.resize(DATASET_PCM_LENGTH);
            target_pcm.resize(DATASET_PCM_LENGTH);
            while(
                source_stream.read(reinterpret_cast<char*>(source_pcm.data()), DATASET_PCM_LENGTH * sizeof(short)) &&
                target_stream.read(reinterpret_cast<char*>(target_pcm.data()), DATASET_PCM_LENGTH * sizeof(short))
            ) {
                float norm_factor;
                // 短时傅里叶变换
                auto source_tuple = lifuren::dataset::audio::pcm_mag_pha_stft(source_pcm, norm_factor);
                features.push_back(std::move(torch::stack({ std::get<0>(source_tuple), std::get<1>(source_tuple) }).squeeze().to(device)));
                auto target_tuple = lifuren::dataset::audio::pcm_mag_pha_stft(target_pcm, norm_factor);
                labels.push_back(std::move(torch::stack({ std::get<0>(target_tuple), std::get<1>(target_tuple) }).squeeze().to(device)));
            }
            source_stream.close();
            target_stream.close();
        }
    ).map(torch::data::transforms::Stack<>());
    return torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(dataset), batch_size);
}

using AudioFileStyleDatasetLoader = std::invoke_result<
    decltype(&lifuren::dataset::loadAudioFileStyleDataset),
    const size_t&,
    const std::string&
>::type;

} // END OF lifuren::dataset

#endif // END OF LFR_HEADER_CV_AUDIO_DATASET_HPP

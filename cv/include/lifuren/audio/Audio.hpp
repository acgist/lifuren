/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * 音频
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_CV_AUDIO_HPP
#define LFR_HEADER_CV_AUDIO_HPP

#include "torch/nn.h"
#include "torch/optim.h"

#include "lifuren/Model.hpp"
#include "lifuren/Client.hpp"
#include "lifuren/audio/AudioDataset.hpp"

namespace lifuren::audio {

/**
 * 音频推理配置
 */
struct AudioParams {

    std::string model;  // 模型文件
    std::string audio;  // 音频文件
    std::string output; // 输出文件
    
};

using AudioModelClient = ModelClient<lifuren::config::ModelParams, AudioParams, std::string>;

template<typename M>
using AudioModelImplClient = ModelImplClient<lifuren::config::ModelParams, AudioParams, std::string, M>;

/**
 * 作曲终端
 */
template<typename M>
class AudioClient : public AudioModelImplClient<M> {

public:
    std::tuple<bool, std::string> pred(const AudioParams& input) override;

};

template<typename M>
using ComposeClient = AudioClient<M>;

extern std::unique_ptr<lifuren::audio::AudioModelClient> getAudioClient(const std::string& client);

/**
 * 师旷模型
 */
class ShikuangModuleImpl : public torch::nn::Module {

private:
    // 卷积->卷积->GRU GRU 还原->还原
    torch::nn::Conv2d downsample{ nullptr };
    torch::nn::BatchNorm2d norm1{ nullptr };
    torch::nn::BatchNorm2d norm2{ nullptr };
    torch::nn::BatchNorm2d norm3{ nullptr };
    torch::nn::Linear upsample1 { nullptr };
    torch::nn::Linear upsample2 { nullptr };
    torch::nn::Linear upsample3 { nullptr };
    torch::nn::Linear upsample4 { nullptr };
    torch::nn::Linear upsample5 { nullptr };
    torch::nn::Linear upsample6 { nullptr };

public:
    ShikuangModuleImpl();
    virtual ~ShikuangModuleImpl();

public:
    torch::Tensor forward(torch::Tensor input);

};

TORCH_MODULE(ShikuangModule);

/**
 * 师旷模型
 */
class ShikuangModel : public lifuren::Model<
    lifuren::dataset::FileDatasetLoader,
    // torch::nn::L1Loss,
    torch::nn::MSELoss,
    torch::optim::SGD,
    // torch::optim::Adam,
    ShikuangModule
> {

public:
    ShikuangModel(lifuren::config::ModelParams params = {});
    virtual ~ShikuangModel();

public:
    bool defineDataset() override;
    void logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) override;

};

} // END OF lifuren::audio

#endif // END OF LFR_HEADER_CV_AUDIO_HPP

/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * 音频模型
 * 
 * https://pytorch.org/docs/stable/generated/torch.stft.html
 * https://pytorch.org/docs/stable/generated/torch.istft.html
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_CV_AUDIO_MODEL_HPP
#define LFR_HEADER_CV_AUDIO_MODEL_HPP

#include "torch/nn.h"
#include "torch/optim.h"

#include "lifuren/Model.hpp"
#include "lifuren/audio/AudioDataset.hpp"

namespace lifuren::audio {

/**
 * 师旷模型实现
 * 
 * 1. 分类信号
 * 2. 处理信号（人声）
 * 3. 合并信号
 */
class ShikuangModuleImpl : public torch::nn::Module {

private:
    // 卷积->卷积->GRU GRU 还原->还原
    torch::nn::Linear linear1 { nullptr };
    torch::nn::Linear linear2 { nullptr };
    torch::nn::Linear linear3 { nullptr };
    torch::nn::GRU gru1 { nullptr };
    torch::nn::GRU gru2 { nullptr };
    torch::nn::GRU gru3 { nullptr };

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

#endif // END OF LFR_HEADER_CV_AUDIO_MODEL_HPP

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
 * TODO: stft loss
 * 
 * https://pytorch.org/docs/stable/generated/torch.stft.html
 * https://pytorch.org/docs/stable/generated/torch.istft.html
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_CORE_AUDIO_MODEL_HPP
#define LFR_HEADER_CORE_AUDIO_MODEL_HPP

#include "torch/optim.h"

#include "lifuren/Model.hpp"

namespace lifuren::audio {

/**
 * 巴赫模型（音频识谱）
 */
class BachModuleImpl : public torch::nn::Module {

private:
    // TODO: 定义结构

public:
    BachModuleImpl();
    ~BachModuleImpl();

public:
    torch::Tensor forward(torch::Tensor input);

};

TORCH_MODULE(BachModule);

/**
 * 巴赫模型（音频识谱）
 */
class BachModel : public lifuren::Model<torch::nn::MSELoss, torch::optim::SGD, lifuren::audio::BachModule> {

public:
    BachModel(lifuren::config::ModelParams params = {});
    ~BachModel();
    
public:
    void defineDataset() override;
    void logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) override;

};

/**
 * 师旷模型（音频风格迁移）
 */
class ShikuangModuleImpl : public torch::nn::Module {

private:
    torch::nn::BatchNorm2d norm { nullptr };
    torch::nn::Conv1d conv1 { nullptr };
    torch::nn::Conv1d conv2 { nullptr };
    torch::nn::ConvTranspose1d convt1 { nullptr };
    torch::nn::ConvTranspose1d convt2 { nullptr };
    torch::nn::Linear linear1 { nullptr };

public:
    ShikuangModuleImpl();
    ~ShikuangModuleImpl();

public:
    torch::Tensor forward(torch::Tensor input);

};

TORCH_MODULE(ShikuangModule);

/**
 * 师旷模型（音频风格迁移）
 */
class ShikuangModel : public lifuren::Model<torch::nn::MSELoss, torch::optim::SGD, lifuren::audio::ShikuangModule> {

public:
    ShikuangModel(lifuren::config::ModelParams params = {});
    ~ShikuangModel();

public:
    void defineDataset() override;
    void logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) override;

};

/**
 * 贝多芬模型（乐谱钢琴指法）
 */
class BeethovenModuleImpl : public torch::nn::Module {

private:
    // TODO: 定义结构

public:
    BeethovenModuleImpl();
    ~BeethovenModuleImpl();

public:
    torch::Tensor forward(torch::Tensor input);

};

TORCH_MODULE(BeethovenModule);

/**
 * 贝多芬模型（乐谱钢琴指法）
 */
class BeethovenModel : public lifuren::Model<torch::nn::MSELoss, torch::optim::SGD, lifuren::audio::BeethovenModule> {

public:
    BeethovenModel(lifuren::config::ModelParams params = {});
    ~BeethovenModel();
    
public:
    void defineDataset() override;
    void logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) override;

};

} // END OF lifuren::audio

#endif // END OF LFR_HEADER_CORE_AUDIO_MODEL_HPP

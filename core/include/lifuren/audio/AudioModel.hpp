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
#ifndef LFR_HEADER_CORE_AUDIO_MODEL_HPP
#define LFR_HEADER_CORE_AUDIO_MODEL_HPP

#include "torch/nn.h"
#include "torch/optim.h"

#include "lifuren/Model.hpp"
#include "lifuren/Dataset.hpp"

namespace lifuren::audio {

class BachModuleImpl : public torch::nn::Module {

private:

public:
    BachModuleImpl();
    virtual ~BachModuleImpl();

public:
    torch::Tensor forward(torch::Tensor input);

};

TORCH_MODULE(BachModule);

class BachModel : public lifuren::Model<
    lifuren::dataset::FileDatasetLoader,
    torch::nn::MSELoss,
    torch::optim::SGD,
    BachModule
> {

public:
    BachModel(lifuren::config::ModelParams params = {});
    virtual ~BachModel();
    
public:
    bool defineDataset() override;
    void logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) override;

};

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
    virtual ~ShikuangModuleImpl();

public:
    torch::Tensor forward(torch::Tensor input);

};

TORCH_MODULE(ShikuangModule);

class ShikuangModel : public lifuren::Model<
    lifuren::dataset::FileDatasetLoader,
    torch::nn::MSELoss,
    torch::optim::SGD,
    // torch::optim::AdamW,
    ShikuangModule
> {

public:
    ShikuangModel(lifuren::config::ModelParams params = {});
    virtual ~ShikuangModel();

    // TODO: stft loss

public:
    bool defineDataset() override;
    void logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) override;

};

class BeethovenModuleImpl : public torch::nn::Module {

private:

public:
    BeethovenModuleImpl();
    virtual ~BeethovenModuleImpl();

public:
    torch::Tensor forward(torch::Tensor input);

};

TORCH_MODULE(BeethovenModule);

class BeethovenModel : public lifuren::Model<
    lifuren::dataset::FileDatasetLoader,
    torch::nn::MSELoss,
    torch::optim::SGD,
    BeethovenModule
> {

public:
    BeethovenModel(lifuren::config::ModelParams params = {});
    virtual ~BeethovenModel();
    
public:
    bool defineDataset() override;
    void logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) override;

};

} // END OF lifuren::audio

#endif // END OF LFR_HEADER_CORE_AUDIO_MODEL_HPP

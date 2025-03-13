/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * 图片模型
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_CORE_IMAGE_MODEL_HPP
#define LFR_HEADER_CORE_IMAGE_MODEL_HPP

#include "torch/nn.h"
#include "torch/optim.h"

#include "lifuren/Model.hpp"
#include "lifuren/Dataset.hpp"

namespace lifuren::image {

class ChopinModuleImpl : public torch::nn::Module {

private:

public:
    ChopinModuleImpl();
    virtual ~ChopinModuleImpl();

public:
    torch::Tensor forward(torch::Tensor input);

};

TORCH_MODULE(ChopinModule);

class ChopinModel : public lifuren::Model<
    lifuren::dataset::FileDatasetLoader,
    torch::nn::MSELoss,
    torch::optim::SGD,
    ChopinModule
> {

public:
    ChopinModel(lifuren::config::ModelParams params = {});
    virtual ~ChopinModel();
    
public:
    bool defineDataset() override;
    void logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) override;

};

class MozartModuleImpl : public torch::nn::Module {

private:

public:
    MozartModuleImpl();
    virtual ~MozartModuleImpl();

public:
    torch::Tensor forward(torch::Tensor input);

};

TORCH_MODULE(MozartModule);

class MozartModel : public lifuren::Model<
    lifuren::dataset::FileDatasetLoader,
    torch::nn::MSELoss,
    torch::optim::SGD,
    MozartModule
> {

public:
    MozartModel(lifuren::config::ModelParams params = {});
    virtual ~MozartModel();
    
public:
    bool defineDataset() override;
    void logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) override;

};

class WudaoziModuleImpl : public torch::nn::Module {

private:
    torch::nn::Linear linear { nullptr };
    torch::nn::GRU gru1 { nullptr };
    torch::nn::GRU gru2 { nullptr };
    torch::nn::Conv1d conv1 { nullptr };
    torch::nn::Conv1d conv2 { nullptr };
    torch::nn::ConvTranspose1d convt1 { nullptr };
    torch::nn::ConvTranspose1d convt2 { nullptr };

public:
    WudaoziModuleImpl();
    virtual ~WudaoziModuleImpl();

public:
    torch::Tensor forward(torch::Tensor input);

};

TORCH_MODULE(WudaoziModule);

class WudaoziModel : public lifuren::Model<
    lifuren::dataset::FileDatasetLoader,
    torch::nn::MSELoss,
    torch::optim::Adam,
    WudaoziModule
> {

public:
    WudaoziModel(lifuren::config::ModelParams params = {});
    virtual ~WudaoziModel();

public:
    bool defineDataset() override;
    void logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) override;

};

} // END OF lifuren::image

#endif // END OF LFR_HEADER_CORE_IMAGE_MODEL_HPP

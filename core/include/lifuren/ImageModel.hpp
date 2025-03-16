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

#include "torch/optim.h"

#include "lifuren/Model.hpp"

namespace lifuren::image {

/**
 * 肖邦模型（简谱识谱）
 */
class ChopinModuleImpl : public torch::nn::Module {

private:
    // TODO: 定义结构

public:
    ChopinModuleImpl();
    ~ChopinModuleImpl();

public:
    torch::Tensor forward(torch::Tensor input);

};

TORCH_MODULE(ChopinModule);

/**
 * 肖邦模型（简谱识谱）
 */
class ChopinModel : public lifuren::Model<torch::nn::MSELoss, torch::optim::SGD, lifuren::image::ChopinModule> {

public:
    ChopinModel(lifuren::config::ModelParams params = {});
    ~ChopinModel();
    
public:
    bool defineDataset() override;
    void logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) override;

};

/**
 * 莫扎特模型（五线谱识谱）
 */
class MozartModuleImpl : public torch::nn::Module {

private:
    // TODO: 定义结构

public:
    MozartModuleImpl();
    ~MozartModuleImpl();

public:
    torch::Tensor forward(torch::Tensor input);

};

TORCH_MODULE(MozartModule);

/**
 * 莫扎特模型（五线谱识谱）
 */
class MozartModel : public lifuren::Model<torch::nn::MSELoss, torch::optim::SGD, lifuren::image::MozartModule> {

public:
    MozartModel(lifuren::config::ModelParams params = {});
    ~MozartModel();
    
public:
    bool defineDataset() override;
    void logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) override;

};

/**
 * 吴道子模型（图片风格迁移）
 */
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
    ~WudaoziModuleImpl();

public:
    torch::Tensor forward(torch::Tensor input);

};

TORCH_MODULE(WudaoziModule);

/**
 * 吴道子模型（图片风格迁移）
 */
class WudaoziModel : public lifuren::Model<torch::nn::MSELoss, torch::optim::Adam, lifuren::image::WudaoziModule> {

public:
    WudaoziModel(lifuren::config::ModelParams params = {});
    ~WudaoziModel();

public:
    bool defineDataset() override;
    void logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) override;

};

} // END OF lifuren::image

#endif // END OF LFR_HEADER_CORE_IMAGE_MODEL_HPP

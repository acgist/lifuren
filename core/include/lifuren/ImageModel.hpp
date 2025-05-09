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
 * 吴道子模型（视频风格迁移）
 */
class WudaoziModuleImpl : public torch::nn::Module {

private:
    torch::nn::Sequential feature        { nullptr }; // 特征
    torch::nn::Sequential feature_colour { nullptr }; // 特征色调
    torch::nn::Sequential colour_static  { nullptr }; // 色调静态
    torch::nn::Sequential colour_dynamic { nullptr }; // 色调动态
    torch::nn::Sequential feature_live   { nullptr }; // 特征时空
    torch::nn::Sequential live_static    { nullptr }; // 时空静态
    torch::nn::Sequential live_dynamic   { nullptr }; // 时空动态
    torch::nn::Sequential feature_body   { nullptr }; // 特征体型
    torch::nn::Sequential feature_pose   { nullptr }; // 特征动作
    torch::nn::Sequential embody         { nullptr }; // 具象化

public:
    WudaoziModuleImpl();
    ~WudaoziModuleImpl();

public:
    torch::Tensor forward(torch::Tensor input);

};

TORCH_MODULE(WudaoziModule);

/**
 * 吴道子模型（视频风格迁移）
 */
class WudaoziModel : public lifuren::Model<torch::nn::MSELoss, torch::optim::SGD, lifuren::image::WudaoziModule, lifuren::dataset::SeqDatasetLoader> {

public:
    WudaoziModel(lifuren::config::ModelParams params = {});
    ~WudaoziModel();
    
public:
    void defineDataset() override;
    void logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) override;

};

} // END OF lifuren::image

#endif // END OF LFR_HEADER_CORE_IMAGE_MODEL_HPP

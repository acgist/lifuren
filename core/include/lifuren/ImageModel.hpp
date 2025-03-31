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
 * 肖邦模型（五线谱识谱）
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
 * 肖邦模型（五线谱识谱）
 */
class ChopinModel : public lifuren::Model<torch::nn::MSELoss, torch::optim::SGD, lifuren::image::ChopinModule, lifuren::dataset::RndDatasetLoader> {

public:
    ChopinModel(lifuren::config::ModelParams params = {});
    ~ChopinModel();
    
public:
    void defineDataset() override;
    void logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) override;

};

} // END OF lifuren::image

#endif // END OF LFR_HEADER_CORE_IMAGE_MODEL_HPP

/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * 乐谱模型
 * 
 * https://pytorch.org/docs/stable/generated/torch.stft.html
 * https://pytorch.org/docs/stable/generated/torch.istft.html
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_CORE_SCORE_MODEL_HPP
#define LFR_HEADER_CORE_SCORE_MODEL_HPP

#include "torch/optim.h"

#include "lifuren/Model.hpp"

namespace lifuren::score {


/**
 * 莫扎特模型（乐谱钢琴指法）
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
 * 莫扎特模型（乐谱钢琴指法）
 */
class MozartModel : public lifuren::Model<torch::nn::MSELoss, torch::optim::SGD, lifuren::score::MozartModule> {

public:
    MozartModel(lifuren::config::ModelParams params = {});
    ~MozartModel();
    
public:
    void defineDataset() override;
    void logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) override;

};

} // END OF lifuren::score

#endif // END OF LFR_HEADER_CORE_SCORE_MODEL_HPP

/**
 * 导演模型
 */
#ifndef LFR_HEADER_CV_ACT_MODEL_HPP
#define LFR_HEADER_CV_ACT_MODEL_HPP

#include "lifuren/Model.hpp"
#include "lifuren/video/VideoDataset.hpp"

namespace lifuren {

/**
 * 汤显祖模型
 */
class TangxianzuModuleImpl : public torch::nn::Module {

private:
    // TODO: 模型定义

public:
    TangxianzuModuleImpl();
    virtual ~TangxianzuModuleImpl();

public:
    torch::Tensor forward(torch::Tensor input);

};

TORCH_MODULE(TangxianzuModule);

/**
 * 汤显祖模型
 */
class TangxianzuModel : public lifuren::Model<
    lifuren::dataset::RawDatasetLoader,
    torch::nn::MSELoss,
    torch::optim::Adam,
    TangxianzuModule
> {

public:
    TangxianzuModel(lifuren::config::ModelParams params = {});
    virtual ~TangxianzuModel();

public:
    bool defineDataset() override;
    void logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) override;

};

/**
 * 关汉卿模型
 */
class GuanhanqingModuleImpl : public torch::nn::Module {

private:
    // TODO: 模型定义
    
public:
    GuanhanqingModuleImpl();
    virtual ~GuanhanqingModuleImpl();

public:
    torch::Tensor forward(torch::Tensor input);

};

TORCH_MODULE(GuanhanqingModule);

/**
 * 关汉卿模型
 */
class GuanhanqingModel : public lifuren::Model<
    lifuren::dataset::RawDatasetLoader,
    torch::nn::MSELoss,
    torch::optim::Adam,
    GuanhanqingModule
> {

public:
    GuanhanqingModel(lifuren::config::ModelParams params = {});
    virtual ~GuanhanqingModel();

public:
    bool defineDataset() override;
    void logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) override;

};

}

#endif // END OF LFR_HEADER_CV_ACT_MODEL_HPP

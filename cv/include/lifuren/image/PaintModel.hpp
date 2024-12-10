/**
 * 绘画模型
 */
#ifndef LFR_HEADER_CV_PAINT_MODEL_HPP
#define LFR_HEADER_CV_PAINT_MODEL_HPP

#include "lifuren/Model.hpp"

#include "lifuren/image/ImageDataset.hpp"

namespace lifuren {

/**
 * 吴道子模型
 */
class WudaoziModuleImpl : public torch::nn::Module {

private:
    // TODO: 模型定义

public:
    WudaoziModuleImpl();
    virtual ~WudaoziModuleImpl();

public:
    torch::Tensor forward(torch::Tensor input);

};

TORCH_MODULE(WudaoziModule);

/**
 * 吴道子模型
 */
class WudaoziModel : public lifuren::Model<
    lifuren::dataset::ImageFileStyleDatasetLoader,
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

/**
 * 顾恺之模型
 */
class GukaizhiModuleImpl : public torch::nn::Module {

private:
    // TODO: 模型定义

public:
    GukaizhiModuleImpl();
    virtual ~GukaizhiModuleImpl();

public:
    torch::Tensor forward(torch::Tensor input);

};

TORCH_MODULE(GukaizhiModule);

/**
 * 顾恺之模型
 */
class GukaizhiModel : public lifuren::Model<
    lifuren::dataset::ImageFileGANDatasetLoader,
    torch::nn::MSELoss,
    torch::optim::Adam,
    GukaizhiModule
> {

public:
    GukaizhiModel(lifuren::config::ModelParams params = {});
    virtual ~GukaizhiModel();

public:
    bool defineDataset() override;
    void logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) override;

};

}

#endif // END OF LFR_HEADER_CV_PAINT_MODEL_HPP
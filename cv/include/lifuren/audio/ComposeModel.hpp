/**
 * 作曲模型
 */
#ifndef LFR_HEADER_CV_COMPOSE_DATASET_HPP
#define LFR_HEADER_CV_COMPOSE_DATASET_HPP

#include "lifuren/Model.hpp"
#include "lifuren/audio/AudioDataset.hpp"

namespace lifuren {

/**
 * 师旷模型
 */
class ShikuangModuleImpl : public torch::nn::Module {

private:
    // TODO: 模型定义

// 卷积->卷积->GRU GRU 还原->还原
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
    lifuren::dataset::AudioFileStyleDatasetLoader,
    torch::nn::MSELoss,
    torch::optim::Adam,
    ShikuangModule
> {

public:
    ShikuangModel(lifuren::config::ModelParams params = {});
    virtual ~ShikuangModel();

public:
    bool defineDataset() override;
    void logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) override;

};

/**
 * 李龟年模型
 */
class LiguinianModuleImpl : public torch::nn::Module {

private:
    // TODO: 模型定义

public:
    LiguinianModuleImpl();
    virtual ~LiguinianModuleImpl();

public:
    torch::Tensor forward(torch::Tensor input);

};

TORCH_MODULE(LiguinianModule);

/**
 * 李龟年模型
 */
class LiguinianModel : public lifuren::Model<
    lifuren::dataset::AudioFileGANDatasetLoader,
    torch::nn::MSELoss,
    torch::optim::Adam,
    LiguinianModule
> {

public:
    LiguinianModel(lifuren::config::ModelParams params = {});
    virtual ~LiguinianModel();

public:
    bool defineDataset() override;
    void logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) override;

};

}

#endif // END OF LFR_HEADER_CV_COMPOSE_DATASET_HPP

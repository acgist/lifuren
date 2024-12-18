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
    // 卷积->卷积->GRU GRU 还原->还原
    torch::nn::Conv2d downsample{ nullptr };
    torch::nn::BatchNorm2d norm1{ nullptr };
    torch::nn::BatchNorm2d norm2{ nullptr };
    torch::nn::BatchNorm2d norm3{ nullptr };
    torch::nn::Linear upsample1 { nullptr };
    torch::nn::Linear upsample2 { nullptr };
    torch::nn::Linear upsample3 { nullptr };
    torch::nn::Linear upsample4 { nullptr };
    torch::nn::Linear upsample5 { nullptr };
    torch::nn::Linear upsample6 { nullptr };

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
    // torch::nn::L1Loss,
    torch::nn::MSELoss,
    torch::optim::SGD,
    // torch::optim::Adam,
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

/**
 * 诗词模型
 */
#ifndef LFR_HEADER_NLP_POETIZE_MODEL_HPP
#define LFR_HEADER_NLP_POETIZE_MODEL_HPP

#include "lifuren/Model.hpp"
#include "lifuren/poetry/PoetryDataset.hpp"

namespace lifuren {

/**
 * 李杜模型
 */
class LiduModel {
    // TODO: 实现
};

class SuxinModuleImpl : public torch::nn::Module {

private:
    torch::nn::GRU    gru   { nullptr };
    torch::nn::Linear linear{ nullptr };

public:
    SuxinModuleImpl();
    virtual ~SuxinModuleImpl();

public:
    torch::Tensor forward(torch::Tensor input);

};

TORCH_MODULE(SuxinModule);

/**
 * 苏辛模型
 */
class SuxinModel : public lifuren::Model<
    lifuren::dataset::PoetryFileDatasetLoader,
    torch::nn::MSELoss,
    torch::optim::Adam,
    SuxinModule
> {

public:
    SuxinModel(lifuren::ModelParams params = {});
    virtual ~SuxinModel();

public:
    bool defineDataset() override;
    void logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) override;

};

}

#endif // END OF LFR_HEADER_NLP_POETIZE_MODEL_HPP
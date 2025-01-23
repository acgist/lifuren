/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * 诗词模型
 * 
 * 诗仙
 * 诗圣
 * 诗佛
 * 诗鬼
 * 诗魔
 * 李杜
 * 苏辛
 * 婉约
 * 
 * 模型实现：李杜、苏辛
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_CV_POETRY_MODEL_HPP
#define LFR_HEADER_CV_POETRY_MODEL_HPP

#include "torch/nn.h"
#include "torch/optim.h"

#include "lifuren/File.hpp"
#include "lifuren/Model.hpp"
#include "lifuren/poetry/PoetryDataset.hpp"

namespace lifuren::poetry {

/**
 * 李杜模型
 */
class LiduModuleImpl : public torch::nn::Module {

private:
    // TODO: 模型定义

public:
    LiduModuleImpl();
    virtual ~LiduModuleImpl();

public:
    torch::Tensor forward(torch::Tensor input);

};

TORCH_MODULE(LiduModule);

/**
 * 李杜模型
 */
class LiduModel : public lifuren::Model<
    lifuren::dataset::FileDatasetLoader,
    torch::nn::MSELoss,
    torch::optim::Adam,
    LiduModule
> {

public:
    LiduModel(lifuren::config::ModelParams params = {});
    virtual ~LiduModel();

public:
    bool defineDataset() override;
    void logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) override;

};

/**
 * 苏辛模型
 */
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
    lifuren::dataset::FileDatasetLoader,
    torch::nn::MSELoss,
    torch::optim::Adam,
    SuxinModule
> {

public:
    SuxinModel(lifuren::config::ModelParams params = {});
    virtual ~SuxinModel();

public:
    bool defineDataset() override;
    void logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) override;

};

} // END OF lifuren::poetry

lifuren::poetry::LiduModuleImpl::LiduModuleImpl() {
}

lifuren::poetry::LiduModuleImpl::~LiduModuleImpl() {
}

torch::Tensor lifuren::poetry::LiduModuleImpl::forward(torch::Tensor input) {
    // TODO: 实现
    return input;
}

lifuren::poetry::LiduModel::LiduModel(lifuren::config::ModelParams params) : Model(params) {
}

lifuren::poetry::LiduModel::~LiduModel() {
}

bool lifuren::poetry::LiduModel::defineDataset() {
    if(lifuren::file::exists(this->params.train_path)) {
        this->trainDataset = lifuren::poetry::loadFileDatasetLoader(this->params.batch_size, this->params.train_path);
    }
    if(lifuren::file::exists(this->params.val_path)) {
        this->valDataset = lifuren::poetry::loadFileDatasetLoader(this->params.batch_size, this->params.val_path);
    }
    if(lifuren::file::exists(this->params.test_path)) {
        this->testDataset = lifuren::poetry::loadFileDatasetLoader(this->params.batch_size, this->params.test_path);
    }
    return true;
}

void lifuren::poetry::LiduModel::logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) {
    // TODO: 实现
}

lifuren::poetry::SuxinModuleImpl::SuxinModuleImpl() {
    const auto& poetry = lifuren::config::CONFIG.poetry;
    torch::nn::GRU gru(torch::nn::GRUOptions(poetry.dims, poetry.dims));
    this->gru = register_module("gru", gru);
    torch::nn::Linear linear(torch::nn::LinearOptions(poetry.length + 3, 1));
    this->linear = register_module("linear", linear);
}

lifuren::poetry::SuxinModuleImpl::~SuxinModuleImpl() {
    unregister_module("gru");
    unregister_module("linear");
}

torch::Tensor lifuren::poetry::SuxinModuleImpl::forward(torch::Tensor input) {
    auto [output, hidden] = this->gru->forward(input);
    auto result = this->linear->forward(output.permute({ 2, 1, 0 })).squeeze().t();
    // return torch::log_softmax(result, 1);
    return result;
}

lifuren::poetry::SuxinModel::SuxinModel(lifuren::config::ModelParams params) : Model(params) {
}

lifuren::poetry::SuxinModel::~SuxinModel() {
}

bool lifuren::poetry::SuxinModel::defineDataset() {
    if(lifuren::file::exists(this->params.train_path)) {
        this->trainDataset = lifuren::poetry::loadFileDatasetLoader(this->params.batch_size, this->params.train_path);
    }
    if(lifuren::file::exists(this->params.val_path)) {
        this->valDataset = lifuren::poetry::loadFileDatasetLoader(this->params.batch_size, this->params.val_path);
    }
    if(lifuren::file::exists(this->params.test_path)) {
        this->testDataset = lifuren::poetry::loadFileDatasetLoader(this->params.batch_size, this->params.test_path);
    }
    return true;
}

void lifuren::poetry::SuxinModel::logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) {
    feature = feature.permute({ 1, 0, 2 });
    pred = this->model->forward(feature);
    loss = this->loss(pred, label);
}

#endif // END OF LFR_HEADER_CV_POETRY_MODEL_HPP

#include "lifuren/AudioModel.hpp"

#include "lifuren/File.hpp"

lifuren::audio::ShikuangModuleImpl::ShikuangModuleImpl() {
    this->norm = this->register_module("norm", torch::nn::BatchNorm2d(201));
    this->conv1 = this->register_module("conv1", torch::nn::Conv1d(torch::nn::Conv1dOptions(14, 32, 3)));
    this->conv2 = this->register_module("conv2", torch::nn::Conv1d(torch::nn::Conv1dOptions(32, 64, 3)));
    this->linear1 = this->register_module("linear1", torch::nn::Linear(197, 197));
    this->convt1 = this->register_module("convt1", torch::nn::ConvTranspose1d(torch::nn::ConvTranspose1dOptions(64, 32, 3)));
    this->convt2 = this->register_module("convt2", torch::nn::ConvTranspose1d(torch::nn::ConvTranspose1dOptions(32, 14, 3)));
}

lifuren::audio::ShikuangModuleImpl::~ShikuangModuleImpl() {
    // this->unregister_module("norm");
}

torch::Tensor lifuren::audio::ShikuangModuleImpl::forward(torch::Tensor input) {
    input = this->norm->forward(input);
    input = input.reshape({100, 201, 14});
    input = input.permute({0, 2, 1});
    input = this->conv1->forward(input);
    input = this->conv2->forward(input);
    input = this->linear1(input);
    input = this->convt1->forward(input);
    input = this->convt2->forward(input);
    input = input.permute({0, 2, 1});
    return input.reshape({100, 201, 7, 2});
}

lifuren::audio::ShikuangModel::ShikuangModel(lifuren::config::ModelParams params) : Model(params) {
}

lifuren::audio::ShikuangModel::~ShikuangModel() {
}

void lifuren::audio::ShikuangModel::defineDataset() {
    if(lifuren::file::exists(this->params.train_path)) {
        this->trainDataset = lifuren::dataset::audio::loadShikuangDatasetLoader(this->params.batch_size, this->params.train_path);
    }
    if(lifuren::file::exists(this->params.val_path)) {
        this->valDataset = lifuren::dataset::audio::loadShikuangDatasetLoader(this->params.batch_size, this->params.val_path);
    }
    if(lifuren::file::exists(this->params.test_path)) {
        this->testDataset = lifuren::dataset::audio::loadShikuangDatasetLoader(this->params.batch_size, this->params.test_path);
    }
}

void lifuren::audio::ShikuangModel::logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) {
    // TODO: stft loss
    pred = this->model->forward(feature);
    loss = this->loss->forward(pred, label);
}

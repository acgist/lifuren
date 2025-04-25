#include "lifuren/ImageModel.hpp"

#include "lifuren/File.hpp"

lifuren::image::WudaoziModuleImpl::WudaoziModuleImpl() {
    this->norm = this->register_module("norm", torch::nn::BatchNorm2d(201));
    this->conv1 = this->register_module("conv1", torch::nn::Conv1d(torch::nn::Conv1dOptions(14, 32, 3)));
    this->conv2 = this->register_module("conv2", torch::nn::Conv1d(torch::nn::Conv1dOptions(32, 64, 3)));
    this->linear1 = this->register_module("linear1", torch::nn::Linear(64, 14));
}

lifuren::image::WudaoziModuleImpl::~WudaoziModuleImpl() {
}

torch::Tensor lifuren::image::WudaoziModuleImpl::forward(torch::Tensor input) {
    auto output = this->norm->forward(input);
    return output;
}

lifuren::image::WudaoziModel::WudaoziModel(lifuren::config::ModelParams params) : Model(params) {
}

lifuren::image::WudaoziModel::~WudaoziModel() {
}

void lifuren::image::WudaoziModel::defineDataset() {
    if(lifuren::file::exists(this->params.train_path)) {
        this->trainDataset = lifuren::dataset::image::loadWudaoziDatasetLoader(LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT, this->params.batch_size, this->params.train_path);
    }
    if(lifuren::file::exists(this->params.val_path)) {
        this->valDataset = lifuren::dataset::image::loadWudaoziDatasetLoader(LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT, this->params.batch_size, this->params.val_path);
    }
    if(lifuren::file::exists(this->params.test_path)) {
        this->testDataset = lifuren::dataset::image::loadWudaoziDatasetLoader(LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT, this->params.batch_size, this->params.test_path);
    }
}

void lifuren::image::WudaoziModel::logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) {
    pred = this->model->forward(feature);
    loss = this->loss->forward(pred, label);
}

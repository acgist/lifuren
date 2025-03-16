#include "lifuren/ImageModel.hpp"

#include "lifuren/File.hpp"

lifuren::image::ChopinModuleImpl::ChopinModuleImpl() {
}

lifuren::image::ChopinModuleImpl::~ChopinModuleImpl() {
}

torch::Tensor lifuren::image::ChopinModuleImpl::forward(torch::Tensor input) {
    return {};
}

lifuren::image::ChopinModel::ChopinModel(lifuren::config::ModelParams params) : Model(params) {
}

lifuren::image::ChopinModel::~ChopinModel() {
}

bool lifuren::image::ChopinModel::defineDataset() {
    return true;
}

void lifuren::image::ChopinModel::logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) {

}

lifuren::image::MozartModuleImpl::MozartModuleImpl() {
}

lifuren::image::MozartModuleImpl::~MozartModuleImpl() {
}

torch::Tensor lifuren::image::MozartModuleImpl::forward(torch::Tensor input) {
    return {};
}

lifuren::image::MozartModel::MozartModel(lifuren::config::ModelParams params) : Model(params) {
}

lifuren::image::MozartModel::~MozartModel() {
}

bool lifuren::image::MozartModel::defineDataset() {
    return true;
}

void lifuren::image::MozartModel::logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) {

}

lifuren::image::WudaoziModuleImpl::WudaoziModuleImpl() {
    this->linear = this->register_module("linear", torch::nn::Linear(torch::nn::LinearOptions(640, 640)));
}

lifuren::image::WudaoziModuleImpl::~WudaoziModuleImpl() {
    this->unregister_module("linear");
}

torch::Tensor lifuren::image::WudaoziModuleImpl::forward(torch::Tensor input) {
    return this->linear->forward(input);
}

lifuren::image::WudaoziModel::WudaoziModel(lifuren::config::ModelParams params) : Model(params) {
}

lifuren::image::WudaoziModel::~WudaoziModel() {
}

bool lifuren::image::WudaoziModel::defineDataset() {
    if(lifuren::file::exists(this->params.train_path)) {
        this->trainDataset = lifuren::dataset::image::loadWudaoziDatasetLoader(LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT, this->params.batch_size, this->params.train_path);
    }
    if(lifuren::file::exists(this->params.val_path)) {
        this->valDataset = lifuren::dataset::image::loadWudaoziDatasetLoader(LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT, this->params.batch_size, this->params.val_path);
    }
    if(lifuren::file::exists(this->params.test_path)) {
        this->testDataset = lifuren::dataset::image::loadWudaoziDatasetLoader(LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT, this->params.batch_size, this->params.test_path);
    }
    return true;
}

void lifuren::image::WudaoziModel::logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) {

}

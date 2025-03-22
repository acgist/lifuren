#include "lifuren/ImageModel.hpp"

#include "lifuren/File.hpp"

lifuren::image::ChopinModuleImpl::ChopinModuleImpl() {
}

lifuren::image::ChopinModuleImpl::~ChopinModuleImpl() {
}

torch::Tensor lifuren::image::ChopinModuleImpl::forward(torch::Tensor input) {
    // TODO
    return {};
}

lifuren::image::ChopinModel::ChopinModel(lifuren::config::ModelParams params) : Model(params) {
}

lifuren::image::ChopinModel::~ChopinModel() {
}

void lifuren::image::ChopinModel::defineDataset() {
    if(lifuren::file::exists(this->params.train_path)) {
        this->trainDataset = lifuren::dataset::image::loadChopinDatasetLoader(LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT, this->params.batch_size, this->params.train_path);
    }
    if(lifuren::file::exists(this->params.val_path)) {
        this->valDataset = lifuren::dataset::image::loadChopinDatasetLoader(LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT, this->params.batch_size, this->params.val_path);
    }
    if(lifuren::file::exists(this->params.test_path)) {
        this->testDataset = lifuren::dataset::image::loadChopinDatasetLoader(LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT, this->params.batch_size, this->params.test_path);
    }
}

void lifuren::image::ChopinModel::logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) {
    pred = this->model->forward(feature);
    loss = this->loss->forward(pred, label);
}

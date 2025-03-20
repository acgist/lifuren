#include "lifuren/ScoreModel.hpp"

#include "lifuren/File.hpp"

lifuren::score::BeethovenModuleImpl::BeethovenModuleImpl() {
}

lifuren::score::BeethovenModuleImpl::~BeethovenModuleImpl() {
}

torch::Tensor lifuren::score::BeethovenModuleImpl::forward(torch::Tensor input) {
    // TODO
    return {};
}

lifuren::score::BeethovenModel::BeethovenModel(lifuren::config::ModelParams params) : Model(params) {
}

lifuren::score::BeethovenModel::~BeethovenModel() {
}

void lifuren::score::BeethovenModel::defineDataset() {
    if(lifuren::file::exists(this->params.train_path)) {
        this->trainDataset = lifuren::dataset::score::loadBeethovenDatasetLoader(this->params.batch_size, this->params.train_path);
    }
    if(lifuren::file::exists(this->params.val_path)) {
        this->valDataset = lifuren::dataset::score::loadBeethovenDatasetLoader(this->params.batch_size, this->params.val_path);
    }
    if(lifuren::file::exists(this->params.test_path)) {
        this->testDataset = lifuren::dataset::score::loadBeethovenDatasetLoader(this->params.batch_size, this->params.test_path);
    }
}

void lifuren::score::BeethovenModel::logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) {
    pred = this->model->forward(feature);
    loss = this->loss->forward(pred, label);
}

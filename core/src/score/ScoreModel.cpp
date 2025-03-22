#include "lifuren/ScoreModel.hpp"

#include "lifuren/File.hpp"

lifuren::score::MozartModuleImpl::MozartModuleImpl() {
}

lifuren::score::MozartModuleImpl::~MozartModuleImpl() {
}

torch::Tensor lifuren::score::MozartModuleImpl::forward(torch::Tensor input) {
    // TODO
    return {};
}

lifuren::score::MozartModel::MozartModel(lifuren::config::ModelParams params) : Model(params) {
}

lifuren::score::MozartModel::~MozartModel() {
}

void lifuren::score::MozartModel::defineDataset() {
    if(lifuren::file::exists(this->params.train_path)) {
        this->trainDataset = lifuren::dataset::score::loadMozartDatasetLoader(this->params.batch_size, this->params.train_path);
    }
    if(lifuren::file::exists(this->params.val_path)) {
        this->valDataset = lifuren::dataset::score::loadMozartDatasetLoader(this->params.batch_size, this->params.val_path);
    }
    if(lifuren::file::exists(this->params.test_path)) {
        this->testDataset = lifuren::dataset::score::loadMozartDatasetLoader(this->params.batch_size, this->params.test_path);
    }
}

void lifuren::score::MozartModel::logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) {
    pred = this->model->forward(feature);
    loss = this->loss->forward(pred, label);
}

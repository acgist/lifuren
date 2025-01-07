#include "lifuren/poetry/PoetryModel.hpp"

#include "lifuren/File.hpp"

lifuren::LiduModuleImpl::LiduModuleImpl() {
}

lifuren::LiduModuleImpl::~LiduModuleImpl() {
}

torch::Tensor lifuren::LiduModuleImpl::forward(torch::Tensor input) {
    // TODO: 实现
    return input;
}

lifuren::LiduModel::LiduModel(lifuren::config::ModelParams params) : Model(params) {
}

lifuren::LiduModel::~LiduModel() {
}

bool lifuren::LiduModel::defineDataset() {
    if(lifuren::file::exists(this->params.train_path)) {
        this->trainDataset = lifuren::dataset::loadPoetryFileGANDataset(this->params.batch_size, this->params.train_path);
    }
    if(lifuren::file::exists(this->params.val_path)) {
        this->valDataset = lifuren::dataset::loadPoetryFileGANDataset(this->params.batch_size, this->params.val_path);
    }
    if(lifuren::file::exists(this->params.test_path)) {
        this->testDataset = lifuren::dataset::loadPoetryFileGANDataset(this->params.batch_size, this->params.test_path);
    }
    return true;
}

void lifuren::LiduModel::logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) {
    // TODO: 实现
}

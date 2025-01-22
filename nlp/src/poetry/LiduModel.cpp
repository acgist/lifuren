#include "lifuren/poetry/Poetry.hpp"

#include "lifuren/File.hpp"

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

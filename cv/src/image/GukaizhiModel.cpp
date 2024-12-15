#include "lifuren/image/PaintModel.hpp"

#include "lifuren/File.hpp"

#ifndef IMAGE_GAN_WIDTH
#define IMAGE_GAN_WIDTH 640
#endif
#ifndef IMAGE_GAN_HEIGHT
#define IMAGE_GAN_HEIGHT 640
#endif

lifuren::GukaizhiModuleImpl::GukaizhiModuleImpl() {
}

lifuren::GukaizhiModuleImpl::~GukaizhiModuleImpl() {
}

torch::Tensor lifuren::GukaizhiModuleImpl::forward(torch::Tensor input) {
    // TODO: 实现
    return input;
}

lifuren::GukaizhiModel::GukaizhiModel(lifuren::config::ModelParams params) : Model(params) {
}

lifuren::GukaizhiModel::~GukaizhiModel() {
}

bool lifuren::GukaizhiModel::defineDataset() {
    if(lifuren::file::exists(this->params.train_path)) {
        this->trainDataset = lifuren::dataset::loadImageFileGANDataset(IMAGE_GAN_WIDTH, IMAGE_GAN_HEIGHT, this->params.batch_size, this->params.train_path);
    }
    if(lifuren::file::exists(this->params.val_path)) {
        this->valDataset = lifuren::dataset::loadImageFileGANDataset(IMAGE_GAN_WIDTH, IMAGE_GAN_HEIGHT, this->params.batch_size, this->params.val_path);
    }
    if(lifuren::file::exists(this->params.test_path)) {
        this->testDataset = lifuren::dataset::loadImageFileGANDataset(IMAGE_GAN_WIDTH, IMAGE_GAN_HEIGHT, this->params.batch_size, this->params.test_path);
    }
    return true;
}

void lifuren::GukaizhiModel::logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) {
    // TODO: 实现
}

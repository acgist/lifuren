#include "lifuren/video/ActModel.hpp"

#include "lifuren/File.hpp"

#ifndef VIDEO_GAN_WIDTH
#define VIDEO_GAN_WIDTH 640
#endif
#ifndef VIDEO_GAN_HEIGHT
#define VIDEO_GAN_HEIGHT 640
#endif

lifuren::TangxianzuModuleImpl::TangxianzuModuleImpl() {
}

lifuren::TangxianzuModuleImpl::~TangxianzuModuleImpl() {
}

torch::Tensor lifuren::TangxianzuModuleImpl::forward(torch::Tensor input) {
    // TODO: 实现
    return input;
}

lifuren::TangxianzuModel::TangxianzuModel(lifuren::config::ModelParams params) : Model(params) {
}

lifuren::TangxianzuModel::~TangxianzuModel() {
}

bool lifuren::TangxianzuModel::defineDataset() {
    if(lifuren::file::exists(this->params.train_path)) {
        this->trainDataset = lifuren::dataset::loadVideoFileGANDataset(VIDEO_GAN_WIDTH, VIDEO_GAN_HEIGHT, this->params.batch_size, this->params.train_path);
    }
    if(lifuren::file::exists(this->params.val_path)) {
        this->valDataset = lifuren::dataset::loadVideoFileGANDataset(VIDEO_GAN_WIDTH, VIDEO_GAN_HEIGHT, this->params.batch_size, this->params.val_path);
    }
    if(lifuren::file::exists(this->params.test_path)) {
        this->testDataset = lifuren::dataset::loadVideoFileGANDataset(VIDEO_GAN_WIDTH, VIDEO_GAN_HEIGHT, this->params.batch_size, this->params.test_path);
    }
    return true;
}

void lifuren::TangxianzuModel::logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) {
    // TODO: 实现
}

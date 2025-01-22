#include "lifuren/video/Video.hpp"

#include "lifuren/File.hpp"

#ifndef VIDEO_GAN_WIDTH
#define VIDEO_GAN_WIDTH 640
#endif
#ifndef VIDEO_GAN_HEIGHT
#define VIDEO_GAN_HEIGHT 640
#endif

lifuren::video::WudaoziModuleImpl::WudaoziModuleImpl() {
}

lifuren::video::WudaoziModuleImpl::~WudaoziModuleImpl() {
}

torch::Tensor lifuren::video::WudaoziModuleImpl::forward(torch::Tensor input) {
    // TODO: 实现
    return input;
}

lifuren::video::WudaoziModel::WudaoziModel(lifuren::config::ModelParams params) : Model(params) {
}

lifuren::video::WudaoziModel::~WudaoziModel() {
}

bool lifuren::video::WudaoziModel::defineDataset() {
    if(lifuren::file::exists(this->params.train_path)) {
        this->trainDataset = lifuren::video::loadFileDatasetLoader(VIDEO_GAN_WIDTH, VIDEO_GAN_HEIGHT, this->params.batch_size, this->params.train_path);
    }
    if(lifuren::file::exists(this->params.val_path)) {
        this->valDataset = lifuren::video::loadFileDatasetLoader(VIDEO_GAN_WIDTH, VIDEO_GAN_HEIGHT, this->params.batch_size, this->params.val_path);
    }
    if(lifuren::file::exists(this->params.test_path)) {
        this->testDataset = lifuren::video::loadFileDatasetLoader(VIDEO_GAN_WIDTH, VIDEO_GAN_HEIGHT, this->params.batch_size, this->params.test_path);
    }
    return true;
}

void lifuren::video::WudaoziModel::logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) {
    // TODO: 实现
}

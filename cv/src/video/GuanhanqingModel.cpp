#include "lifuren/video/ActModel.hpp"

#include "lifuren/File.hpp"

#ifndef VIDEO_STYLE_WIDTH
#define VIDEO_STYLE_WIDTH 640
#endif
#ifndef VIDEO_STYLE_HEIGHT
#define VIDEO_STYLE_HEIGHT 640
#endif

lifuren::GuanhanqingModuleImpl::GuanhanqingModuleImpl() {
}

lifuren::GuanhanqingModuleImpl::~GuanhanqingModuleImpl() {
}

torch::Tensor lifuren::GuanhanqingModuleImpl::forward(torch::Tensor input) {
    // TODO: 实现
    return input;
}

lifuren::GuanhanqingModel::GuanhanqingModel(lifuren::config::ModelParams params) : Model(params) {
}

lifuren::GuanhanqingModel::~GuanhanqingModel() {
}

bool lifuren::GuanhanqingModel::defineDataset() {
    if(lifuren::file::exists(this->params.train_path)) {
        this->trainDataset = lifuren::dataset::loadVideoFileStyleDataset(VIDEO_STYLE_WIDTH, VIDEO_STYLE_HEIGHT, this->params.batch_size, this->params.train_path);
    }
    if(lifuren::file::exists(this->params.val_path)) {
        this->valDataset = lifuren::dataset::loadVideoFileStyleDataset(VIDEO_STYLE_WIDTH, VIDEO_STYLE_HEIGHT, this->params.batch_size, this->params.val_path);
    }
    if(lifuren::file::exists(this->params.test_path)) {
        this->testDataset = lifuren::dataset::loadVideoFileStyleDataset(VIDEO_STYLE_WIDTH, VIDEO_STYLE_HEIGHT, this->params.batch_size, this->params.test_path);
    }
    return true;
}

void lifuren::GuanhanqingModel::logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) {
    // TODO: 实现
}

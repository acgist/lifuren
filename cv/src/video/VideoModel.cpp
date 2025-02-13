#include "lifuren/video/VideoModel.hpp"

#include "lifuren/File.hpp"

lifuren::video::WudaoziModuleImpl::WudaoziModuleImpl() {
    this->linear = this->register_module("linear", torch::nn::Linear(torch::nn::LinearOptions(640, 640)));
}

lifuren::video::WudaoziModuleImpl::~WudaoziModuleImpl() {
    this->unregister_module("linear");
}

torch::Tensor lifuren::video::WudaoziModuleImpl::forward(torch::Tensor input) {
    return this->linear->forward(input);
}

lifuren::video::WudaoziModel::WudaoziModel(lifuren::config::ModelParams params) : Model(params) {
}

lifuren::video::WudaoziModel::~WudaoziModel() {
}

bool lifuren::video::WudaoziModel::defineDataset() {
    if(lifuren::file::exists(this->params.train_path)) {
        this->trainDataset = lifuren::video::loadFileDatasetLoader(LFR_VIDEO_WIDTH, LFR_VIDEO_HEIGHT, this->params.batch_size, this->params.train_path);
    }
    if(lifuren::file::exists(this->params.val_path)) {
        this->valDataset = lifuren::video::loadFileDatasetLoader(LFR_VIDEO_WIDTH, LFR_VIDEO_HEIGHT, this->params.batch_size, this->params.val_path);
    }
    if(lifuren::file::exists(this->params.test_path)) {
        this->testDataset = lifuren::video::loadFileDatasetLoader(LFR_VIDEO_WIDTH, LFR_VIDEO_HEIGHT, this->params.batch_size, this->params.test_path);
    }
    return true;
}

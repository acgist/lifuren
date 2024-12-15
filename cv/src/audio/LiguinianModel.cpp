#include "lifuren/audio/ComposeModel.hpp"

#include "lifuren/File.hpp"

#ifndef AUDIO_GAN_LENGTH
#define AUDIO_GAN_LENGTH 48000 * 60
#endif

lifuren::LiguinianModuleImpl::LiguinianModuleImpl() {
}

lifuren::LiguinianModuleImpl::~LiguinianModuleImpl() {
}

torch::Tensor lifuren::LiguinianModuleImpl::forward(torch::Tensor input) {
    // TODO: 实现
    return input;
}

lifuren::LiguinianModel::LiguinianModel(lifuren::config::ModelParams params) : Model(params) {
}

lifuren::LiguinianModel::~LiguinianModel() {
}

bool lifuren::LiguinianModel::defineDataset() {
    if(lifuren::file::exists(this->params.train_path)) {
        this->trainDataset = lifuren::dataset::loadAudioFileGANDataset(AUDIO_GAN_LENGTH, this->params.batch_size, this->params.train_path);
    }
    if(lifuren::file::exists(this->params.val_path)) {
        this->valDataset = lifuren::dataset::loadAudioFileGANDataset(AUDIO_GAN_LENGTH, this->params.batch_size, this->params.val_path);
    }
    if(lifuren::file::exists(this->params.test_path)) {
        this->testDataset = lifuren::dataset::loadAudioFileGANDataset(AUDIO_GAN_LENGTH, this->params.batch_size, this->params.test_path);
    }
    return true;
}

void lifuren::LiguinianModel::logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) {
    // TODO: 实现
}

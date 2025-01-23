/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * 视频模型
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_CV_VIDEO_MODEL_HPP
#define LFR_HEADER_CV_VIDEO_MODEL_HPP

#include "torch/nn.h"
#include "torch/optim.h"

#include "lifuren/File.hpp"
#include "lifuren/Model.hpp"
#include "lifuren/video/VideoDataset.hpp"

#ifndef VIDEO_GAN_WIDTH
#define VIDEO_GAN_WIDTH 640
#endif
#ifndef VIDEO_GAN_HEIGHT
#define VIDEO_GAN_HEIGHT 640
#endif

namespace lifuren::video {

/**
 * 吴道子模型
 */
class WudaoziModuleImpl : public torch::nn::Module {

private:
    // TODO: 模型定义

public:
    WudaoziModuleImpl();
    virtual ~WudaoziModuleImpl();

public:
    torch::Tensor forward(torch::Tensor input);

};

TORCH_MODULE(WudaoziModule);

/**
 * 吴道子模型
 */
class WudaoziModel : public lifuren::Model<
    lifuren::dataset::FileDatasetLoader,
    torch::nn::MSELoss,
    torch::optim::Adam,
    WudaoziModule
> {

public:
    WudaoziModel(lifuren::config::ModelParams params = {});
    virtual ~WudaoziModel();

public:
    bool defineDataset() override;
    void logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) override;

};

} // END OF lifuren::video

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

#endif // END OF LFR_HEADER_CV_VIDEO_MODEL_HPP

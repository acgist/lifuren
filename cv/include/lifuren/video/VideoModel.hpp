/**
 * 视频模型
 */
#ifndef LFR_HEADER_CV_VIDEO_MODEL_HPP
#define LFR_HEADER_CV_VIDEO_MODEL_HPP

#include "torch/nn.h"
#include "torch/optim.h"

#include "lifuren/Model.hpp"
#include "lifuren/video/VideoDataset.hpp"

namespace lifuren {

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
    lifuren::dataset::VideoFileGANDatasetLoader,
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

}

#endif // END OF LFR_HEADER_CV_VIDEO_MODEL_HPP

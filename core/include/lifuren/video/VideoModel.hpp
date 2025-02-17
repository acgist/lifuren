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

#include "lifuren/Model.hpp"
#include "lifuren/video/VideoDataset.hpp"

// 视频配置
#ifndef LFR_VIDEO_CONFIG
#define LFR_VIDEO_CONFIG
#define LFR_VIDEO_FPS      24  // 视频帧率
#define LFR_VIDEO_WIDTH    640 // 视频宽度
#define LFR_VIDEO_HEIGHT   480 // 视频高度
#define LFR_VIDEO_PRED_FPS 24  // 推理帧数
#endif

namespace lifuren::video {

/**
 * 吴道子模型实现
 */
class WudaoziModuleImpl : public torch::nn::Module {

private:
    torch::nn::Linear linear { nullptr };

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

};

} // END OF lifuren::video

#endif // END OF LFR_HEADER_CV_VIDEO_MODEL_HPP

/**
 * 视频工具
 */
#ifndef LFR_HEADER_CV_VIDEO_HPP
#define LFR_HEADER_CV_VIDEO_HPP

#include "torch/nn.h"
#include "torch/optim.h"

#include "lifuren/Model.hpp"
#include "lifuren/Client.hpp"
#include "lifuren/video/VideoDataset.hpp"

namespace lifuren::video {

/**
 * 视频推理配置
 */
struct VideoParams {

    std::string model;  // 模型文件
    std::string video;  // 视频文件
    std::string output; // 输出文件
    
};

using VideoModelClient = ModelClient<lifuren::config::ModelParams, VideoParams, std::string>;

template<typename M>
using VideoModelImplClient = ModelImplClient<lifuren::config::ModelParams, VideoParams, std::string, M>;

/**
 * 视频终端
 */
template<typename M>
class VideoClient : public VideoModelImplClient<M> {

public:
    std::tuple<bool, std::string> pred(const VideoParams& input) override;

};

template<typename M>
using PaintClient = VideoClient<M>;

extern std::unique_ptr<lifuren::video::VideoModelClient> getVideoClient(const std::string& client);

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

}

#endif // END OF LFR_HEADER_CV_VIDEO_HPP

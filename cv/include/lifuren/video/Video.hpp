/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * 视频
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_CV_VIDEO_HPP
#define LFR_HEADER_CV_VIDEO_HPP

#include "lifuren/Client.hpp"

namespace lifuren::video {

/**
 * 视频推理配置
 */
struct VideoParams {

    std::string video; // 图片文件（.jpg/.png/.jpeg） | 视频文件（.mp4）
    
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

/**
 * @return 视频终端
 */
extern std::unique_ptr<lifuren::video::VideoModelClient> getVideoClient(
    const std::string& model // 模型名称
);

}

#endif // END OF LFR_HEADER_CV_VIDEO_HPP

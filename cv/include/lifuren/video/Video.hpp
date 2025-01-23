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

}

#endif // END OF LFR_HEADER_CV_VIDEO_HPP

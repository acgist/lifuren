/**
 * 视频终端
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CV_VIDEO_CLIENT_HPP
#define LFR_HEADER_CV_VIDEO_CLIENT_HPP

#include "lifuren/Client.hpp"

namespace lifuren {

struct VideoParams {

    std::string model;  // 模型
    std::string video;  // 视频文件
    std::string output; // 输出文件
    
};

using VideoModelClient = ModelClient<lifuren::config::ModelParams, VideoParams, std::string>;

template<typename M>
using VideoModelImplClient = ModelImplClient<lifuren::config::ModelParams, VideoParams, std::string, M>;

extern std::unique_ptr<lifuren::VideoModelClient> getVideoClient(const std::string& client);

/**
 * 视频终端
 */
template<typename M>
class VideoClient : public VideoModelImplClient<M> {

public:
    VideoClient();
    virtual ~VideoClient();

public:
    std::tuple<bool, std::string> pred(const VideoParams& input) override;

};

template<typename M>
using PaintClient = VideoClient<M>;

} // END OF lifuren

#endif // END OF LFR_HEADER_CV_VIDEO_CLIENT_HPP

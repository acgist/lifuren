#include "lifuren/video/Video.hpp"

#include "lifuren/video/VideoModel.hpp"

template<typename M>
std::tuple<bool, std::string> lifuren::video::VideoClient<M>::pred(const VideoParams& input) {
    // TODO: 实现
    return {};
};

std::unique_ptr<lifuren::video::VideoModelClient> lifuren::video::getVideoClient(const std::string& client) {
    if(client == lifuren::config::CONFIG_VIDEO_WUDAOZI) {
        return std::make_unique<lifuren::video::VideoClient<WudaoziModel>>();
    } else {
        return nullptr;
    }
}

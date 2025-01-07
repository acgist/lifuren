#include "lifuren/video/VideoClient.hpp"

#include "lifuren/video/VideoModel.hpp"

template<typename M>
lifuren::VideoClient<M>::VideoClient() {
}

template<typename M>
lifuren::VideoClient<M>::~VideoClient() {
}

template<typename M>
std::string lifuren::VideoClient<M>::pred(const VideoParams& input) {
    // TODO: 实现
    return {};
};

std::unique_ptr<lifuren::VideoModelClient> lifuren::getVideoClient(const std::string& client) {
    if(client == lifuren::config::CONFIG_VIDEO_WUDAOZI) {
        return std::make_unique<lifuren::VideoClient<WudaoziModel>>();
    } else {
        return nullptr;
    }
}

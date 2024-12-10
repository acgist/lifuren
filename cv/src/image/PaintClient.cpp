#include "lifuren/image/PaintClient.hpp"

#include "lifuren/image/PaintModel.hpp"

template<typename M>
lifuren::PaintClient<M>::PaintClient() {
}

template<typename M>
lifuren::PaintClient<M>::~PaintClient() {
}

template<typename M>
std::string lifuren::PaintClient<M>::pred(const PaintParams& input) {
    // TODO: 实现
    return {};
};

std::unique_ptr<lifuren::PaintModelClient> lifuren::getPaintClient(const std::string& client) {
    if(client == lifuren::config::CONFIG_PAINT_WUDAOZI) {
        // return std::make_unique<lifuren::PaintClient<WudaoziModel>>();
    } else if(client == lifuren::config::CONFIG_PAINT_GUKAIZHI) {
        // return std::make_unique<lifuren::PaintClient<GukaizhiModel>>();
    } else {
        return nullptr;
    }
    return nullptr;
}

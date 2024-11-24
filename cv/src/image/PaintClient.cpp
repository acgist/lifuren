#include "lifuren/image/PaintClient.hpp"

#include "lifuren/image/PaintModel.hpp"

template<typename M>
lifuren::PaintClient<M>::PaintClient() {
}

template<typename M>
lifuren::PaintClient<M>::~PaintClient() {
}

template<typename M>
std::string lifuren::PaintClient<M>::pred(const PaintOptions& input) {
    // TODO: 实现
    return {};
};

template<typename M>
void lifuren::PaintClient<M>::pred(const PaintOptions& input, PaintModelClient::Callback callback) {
    // TODO: 实现
};

std::unique_ptr<lifuren::PaintModelClient> lifuren::getPaintClient(const std::string& client) {
    if(client == lifuren::config::CONFIG_PAINT_CYCLE_GAN) {
        // return std::make_unique<lifuren::PaintClient<CycleGANModel>>();
    } else if(client == lifuren::config::CONFIG_PAINT_STYLE_GAN) {
        // return std::make_unique<lifuren::PaintClient<StyleGANModel>>();
    } else {
        return nullptr;
    }
    return nullptr;
}

#include "lifuren/PaintClient.hpp"

lifuren::PaintClient::PaintClient(lifuren::PaintClient::PaintCallback callback) : callback(callback) {
}

lifuren::PaintClient::~PaintClient() {
}

std::unique_ptr<lifuren::PaintClient> lifuren::PaintClient::getClient(const std::string& client) {
    if(client == lifuren::config::CONFIG_PAINT_CYCLE_GAN) {
        return std::make_unique<lifuren::CycleGANPaintClient>();
    } else if(client == lifuren::config::CONFIG_PAINT_STYLE_GAN) {
        return std::make_unique<lifuren::StyleGANPaintClient>();
    } else {
        return nullptr;
    }
}

#include "lifuren/PaintClient.hpp"

lifuren::PaintClient::PaintClient(lifuren::PaintClient::PaintCallback callback) : callback(callback) {
}

lifuren::PaintClient::~PaintClient() {
}

bool lifuren::PaintClient::release() {
    return true;
}

std::unique_ptr<lifuren::PaintClient> lifuren::PaintClient::getClient(const std::string& client) {
    if(client == "stable-diffusion-cpp") {
        return std::make_unique<lifuren::StableDiffusionCPPPaintClient>();
    } else {
        return nullptr;
    }
}

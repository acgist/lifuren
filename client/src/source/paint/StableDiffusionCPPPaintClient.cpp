#include "lifuren/Client.hpp"

#include "spdlog/spdlog.h"

lifuren::StableDiffusionCPPPaintClient::StableDiffusionCPPPaintClient() {
}

lifuren::StableDiffusionCPPPaintClient::~StableDiffusionCPPPaintClient() {
}

bool lifuren::StableDiffusionCPPPaintClient::paint(const std::string& prompt, lifuren::PaintClient::PaintCallback callback, const std::string& image) {
    if(this->commandClient) {
        SPDLOG_WARN("加载StableDiffusionCPP任务失败：已经存在任务");
        return false;
    }
    this->commandClient = std::make_unique<lifuren::CommandClient>("");
    return true;
}

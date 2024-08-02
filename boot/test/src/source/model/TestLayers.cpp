#include "lifuren/Layers.hpp"

#include "lifuren/Logger.hpp"

#include "spdlog/spdlog.h"

static void testLayers() {
    lifuren::layers::batchNorm2d(2);
    lifuren::layers::layerNorm({ 2 });
    lifuren::layers::instanceNorm2d(2);
    lifuren::layers::groupNorm(2, 2);
    lifuren::layers::conv2d(2, 2, 2);
    lifuren::layers::maxPool2d(2);
    lifuren::layers::avgPool2d(2);
    lifuren::layers::adaptiveAvgPool2d(2);
    lifuren::layers::dropout();
    lifuren::layers::relu();
    lifuren::layers::gru(2, 2);
    lifuren::layers::lstm(2, 2);
}

int main() {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    testLayers();
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}
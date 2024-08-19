#include "lifuren/config/Config.hpp"

#include <sstream>

#include "spdlog/spdlog.h"

#include "lifuren/Logger.hpp"

static void testConfig() {
    auto& config = lifuren::config::CONFIG;
    std::stringstream stream;
    stream << config.toYaml();
    SPDLOG_DEBUG("配置：{}", stream.str());
}

static void testMarkConfig() {
    lifuren::config::ImageMarkConfig config1{
        "1"
    };
    lifuren::config::ImageMarkConfig config2{
        "1"
    };
    SPDLOG_DEBUG("config1.path : {}", config1.path);
    SPDLOG_DEBUG("config2.path : {}", config2.path);
    assert(config1 == "1");
}

int main() {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    testConfig();
    testMarkConfig();
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}

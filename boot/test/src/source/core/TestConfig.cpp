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

int main() {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    testConfig();
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}

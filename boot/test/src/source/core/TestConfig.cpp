#include "Test.hpp"

#include <sstream>

#include "lifuren/Config.hpp"

static void testConfig() {
    auto& config = lifuren::config::CONFIG;
    std::stringstream stream;
    stream << config.toYaml();
    SPDLOG_DEBUG("配置：{}", stream.str());
}

LFR_TEST(
    testConfig();
);

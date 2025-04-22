#include "lifuren/Test.hpp"

#include <sstream>

#include "lifuren/Config.hpp"

[[maybe_unused]] static void testUuid() {
    SPDLOG_DEBUG("uuid : {}", lifuren::config::uuid());
    SPDLOG_DEBUG("uuid : {}", lifuren::config::uuid());
}

[[maybe_unused]] static void testConfig() {
    std::stringstream stream;
    stream << lifuren::config::CONFIG.toYaml();
    SPDLOG_DEBUG("配置：\n{}", stream.str());
}

LFR_TEST(
    testUuid();
    testConfig();
);

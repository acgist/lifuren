#include "lifuren/Test.hpp"

#include "lifuren/Config.hpp"

[[maybe_unused]] static void testUuid() {
    SPDLOG_DEBUG("uuid : {}", lifuren::config::uuid());
    SPDLOG_DEBUG("uuid : {}", lifuren::config::uuid());
}

[[maybe_unused]] static void testConfig() {
    auto config  = lifuren::config::CONFIG.loadFile();
    auto success = lifuren::config::CONFIG.saveFile();
    SPDLOG_DEBUG("wudaozi: {}", config.model_wudaozi);
    assert(success);
}

LFR_TEST(
    testUuid();
    testConfig();
);

#include "lifuren/config/Config.hpp"

#include <sstream>

#include "lifuren/Logger.hpp"

#include "spdlog/spdlog.h"

static void testConfig() {
    auto& configs = lifuren::CONFIGS;
    std::stringstream stream;
    for(auto& v : configs) {
        stream << v.second.toYaml();
        SPDLOG_DEBUG("k = {} | v =\n{}", v.first, stream.str());
        stream.str("");
    }
}

int main() {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    testConfig();
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}

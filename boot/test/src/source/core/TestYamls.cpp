#include "lifuren/Yamls.hpp"

#include "lifuren/Logger.hpp"

#include "spdlog/spdlog.h"

static void testYamls() {
    YAML::Node yaml = lifuren::yamls::loadFile("D:\\tmp\\lifuren.yml");
    YAML::Node node;
    node["lifuren"] = "漂漂亮亮";
    // yaml.push_back(node);
    yaml["lifuren"] = node;
    lifuren::yamls::saveFile(yaml, "D:\\tmp\\lifuren.yml");
}

int main() {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    testYamls();
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}
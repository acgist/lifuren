#include "Test.hpp"

#include "yaml-cpp/yaml.h"

#include "lifuren/Yamls.hpp"

[[maybe_unused]] static void testYamls() {
    YAML::Node yaml = lifuren::yamls::loadFile("D:\\tmp\\lifuren.yml");
    YAML::Node node;
    node["lifuren"] = "漂漂亮亮";
    // yaml.push_back(node);
    yaml["lifuren"] = node;
    lifuren::yamls::saveFile(yaml, "D:\\tmp\\lifuren.yml");
}

LFR_TEST(
    testYamls();
);

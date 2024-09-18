#include "Test.hpp"

#include "yaml-cpp/yaml.h"

#include "lifuren/Files.hpp"
#include "lifuren/Yamls.hpp"

[[maybe_unused]] static void testYamls() {
    YAML::Node yaml = lifuren::yamls::loadFile(lifuren::files::join({lifuren::config::CONFIG.tmp, "lifuren.yml"}).string());
    YAML::Node node;
    node["lifuren"] = "漂漂亮亮";
    // yaml.push_back(node);
    yaml["lifuren"] = node;
    lifuren::yamls::saveFile(yaml, lifuren::files::join({lifuren::config::CONFIG.tmp, "lifuren.yml"}).string());
}

LFR_TEST(
    testYamls();
);

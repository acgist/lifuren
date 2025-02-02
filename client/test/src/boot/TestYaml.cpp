#include "lifuren/Test.hpp"

#include <sstream>

#include "yaml-cpp/yaml.h"

#include "lifuren/File.hpp"
#include "lifuren/Yaml.hpp"

[[maybe_unused]] static void testYaml() {
    YAML::Node yaml = std::move(lifuren::yaml::loadFile(lifuren::file::join({lifuren::config::CONFIG.tmp, "lifuren.yml"}).string()));
    YAML::Node node;
    node["lifuren"] = "漂漂亮亮";
    // yaml.push_back(node);
    yaml["lifuren"] = node;
    lifuren::yaml::saveFile(yaml, lifuren::file::join({lifuren::config::CONFIG.tmp, "lifuren.yml"}).string());
    std::stringstream stream;
    stream << yaml;
    SPDLOG_DEBUG("yaml : \n{}", stream.str());
}

LFR_TEST(
    testYaml();
);

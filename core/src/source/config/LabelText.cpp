#include "../../header/config/Label.hpp"

#include <cstdint>

#include "spdlog/spdlog.h"

lifuren::LabelText::LabelText() {
}

lifuren::LabelText::LabelText(const std::string& name, const YAML::Node& yaml) {
    this->name  = name;
    this->alias = name;
    if(yaml["rhythmic"]) {
        this->rhythmic = yaml["rhythmic"].as<std::string>();
    }
    if(yaml["example"]) {
        this->example = yaml["example"].as<std::string>();
    }
    if(yaml["fontSize"]) {
        this->fontSize = yaml["fontSize"].as<int>();
    }
    if(yaml["segmentSize"]) {
        this->segmentSize = yaml["segmentSize"].as<int>();
    }
    if(yaml["segmentRule"]) {
        this->segmentRule = yaml["segmentRule"].as<std::vector<uint32_t>>();
    }
    if(yaml["participleRule"]) {
        this->participleRule = yaml["participleRule"].as<std::vector<uint32_t>>();
    }
}

YAML::Node lifuren::LabelText::toYaml() {
    YAML::Node yaml;
    yaml["rhythmic"]       = this->rhythmic;
    yaml["example"]        = this->example;
    yaml["fontSize"]       = this->fontSize;
    yaml["segmentSize"]    = this->segmentSize;
    yaml["segmentRule"]    = this->segmentRule;
    yaml["participleRule"] = this->participleRule;
    return yaml;
}

std::map<std::string, lifuren::LabelText> lifuren::LabelText::loadFile(const std::string& path) {
    SPDLOG_DEBUG("加载标签文件：{}", path);
    std::map<std::string, LabelText> map;
    YAML::Node yaml = lifuren::yamls::loadFile(path);
    if(yaml.size() == 0L) {
        return map;
    }
    for(
        auto iterator = yaml.begin();
        iterator != yaml.end();
        ++iterator
    ) {
        std::string key = iterator->first.as<std::string>();
        LabelText label(
            key,
            iterator->second
        );
        map.emplace(key, label);
    }
    return map;
    // 没有必要使用move
    // return std::move(map);
}

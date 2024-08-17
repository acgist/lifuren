#include "lifuren/config/Label.hpp"

#include <cstdint>
#include <sstream>

#include "spdlog/spdlog.h"

#include "lifuren/Yamls.hpp"

#ifndef LFR_LABEL_TEXT_SETTER
#define LFR_LABEL_TEXT_SETTER(source, key, target, field, type) \
const auto& key##Node = source[#key];                           \
if(key##Node) {                                                 \
    target.field = key##Node.as<type>();                        \
}
#endif

lifuren::LabelText::LabelText() {
}

lifuren::LabelText::LabelText(const std::string& name) : Label(name, name) {
}

std::string lifuren::LabelText::toYaml() {
    YAML::Node yaml;
    yaml["rhythmic"]       = this->rhythmic;
    yaml["example"]        = this->example;
    yaml["fontSize"]       = this->fontSize;
    yaml["segmentSize"]    = this->segmentSize;
    yaml["segmentRule"]    = this->segmentRule;
    yaml["participleRule"] = this->participleRule;
    std::stringstream stream;
    stream << yaml;
    return stream.str();
}

std::map<std::string, lifuren::LabelText> lifuren::LabelText::loadFile(const std::string& path) {
    SPDLOG_DEBUG("加载标签文件：{}", path);
    std::map<std::string, LabelText> map;
    YAML::Node yaml = lifuren::yamls::loadFile(path);
    if(yaml.size() == 0L) {
        return map;
    }
    for(auto iterator = yaml.begin(); iterator != yaml.end(); ++iterator) {
        const std::string key = iterator->first.as<std::string>();
        const auto value      = iterator->second;
        LabelText label(key);
        LFR_LABEL_TEXT_SETTER(value, rhythmic,       label, rhythmic,       std::string);
        LFR_LABEL_TEXT_SETTER(value, example,        label, example,        std::string);
        LFR_LABEL_TEXT_SETTER(value, fontSize,       label, fontSize,       int);
        LFR_LABEL_TEXT_SETTER(value, segmentSize,    label, segmentSize,    int);
        LFR_LABEL_TEXT_SETTER(value, segmentRule,    label, segmentRule,    std::vector<uint32_t>);
        LFR_LABEL_TEXT_SETTER(value, participleRule, label, participleRule, std::vector<uint32_t>);
        map.emplace(key, label);
    }
    return map;
}

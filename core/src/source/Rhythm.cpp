#include "lifuren/Config.hpp"

#include <cstdint>
#include <sstream>

#include "spdlog/spdlog.h"

#include "yaml-cpp/yaml.h"

#include "lifuren/Yamls.hpp"

#ifndef LFR_RHYTHM_SETTER
#define LFR_RHYTHM_SETTER(source, key, target, field, type) \
const auto& key##Node = source[#key];                       \
if(key##Node) {                                             \
    target.field = key##Node.as<type>();                    \
}
#endif

std::map<std::string, lifuren::config::Rhythm> lifuren::config::RHYTHM = lifuren::config::Rhythm::loadFile(lifuren::config::RHYTHM_PATH);

lifuren::config::Rhythm::Rhythm() {
}

lifuren::config::Rhythm::Rhythm(const std::string& name, const std::string& alias) : name(name), alias(alias) {
}

lifuren::config::Rhythm::~Rhythm() {
}

std::string lifuren::config::Rhythm::toYaml() {
    YAML::Node yaml;
    yaml["rhythm"]         = this->rhythm;
    yaml["example"]        = this->example;
    yaml["fontSize"]       = this->fontSize;
    yaml["segmentSize"]    = this->segmentSize;
    yaml["segmentRule"]    = this->segmentRule;
    yaml["participleRule"] = this->participleRule;
    std::stringstream stream;
    stream << yaml;
    return stream.str();
}

std::map<std::string, lifuren::config::Rhythm> lifuren::config::Rhythm::loadFile(const std::string& path) {
    SPDLOG_DEBUG("加载标签文件：{}", path);
    std::map<std::string, Rhythm> map;
    YAML::Node yaml = lifuren::yamls::loadFile(path);
    if(yaml.size() == 0L) {
        return map;
    }
    for(auto iterator = yaml.begin(); iterator != yaml.end(); ++iterator) {
        const std::string key = iterator->first.as<std::string>();
        const auto value      = iterator->second;
        Rhythm rhythm(key, key);
        LFR_RHYTHM_SETTER(value, rhythm,         rhythm, rhythm,         std::string);
        LFR_RHYTHM_SETTER(value, example,        rhythm, example,        std::string);
        LFR_RHYTHM_SETTER(value, fontSize,       rhythm, fontSize,       int);
        LFR_RHYTHM_SETTER(value, segmentSize,    rhythm, segmentSize,    int);
        LFR_RHYTHM_SETTER(value, segmentRule,    rhythm, segmentRule,    std::vector<uint32_t>);
        LFR_RHYTHM_SETTER(value, participleRule, rhythm, participleRule, std::vector<uint32_t>);
        map.emplace(key, rhythm);
    }
    return map;
}

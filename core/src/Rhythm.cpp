#include "lifuren/Config.hpp"

#include <cstdint>
#include <sstream>

#include "spdlog/spdlog.h"

#include "yaml-cpp/yaml.h"

#include "lifuren/Yaml.hpp"

#ifndef LFR_RHYTHM_SETTER
#define LFR_RHYTHM_SETTER(source, key, target, field, type) \
const auto& key##Node = source[#key];                       \
if(key##Node) {                                             \
    target.field = key##Node.template as<type>();           \
}
#endif

std::map<std::string, lifuren::config::Rhythm> lifuren::config::RHYTHM{};

lifuren::config::Rhythm::Rhythm(const std::string& rhythm) : rhythm(rhythm) {
}

std::string lifuren::config::Rhythm::toYaml() {
    YAML::Node yaml;
    yaml["rhythm"]         = this->rhythm;
    yaml["alias"]          = this->alias;
    yaml["title"]          = this->title;
    yaml["example"]        = this->example;
    yaml["fontSize"]       = this->fontSize;
    yaml["segmentSize"]    = this->segmentSize;
    yaml["segmentRule"]    = this->segmentRule;
    yaml["participleRule"] = this->participleRule;
    std::stringstream stream;
    stream << yaml;
    return stream.str();
}

std::map<std::string, lifuren::config::Rhythm> lifuren::config::Rhythm::loadFile() {
    const std::string path = lifuren::config::baseFile(lifuren::config::RHYTHM_PATH);
    SPDLOG_DEBUG("加载格律文件：{}", path);
    std::map<std::string, Rhythm> map;
    YAML::Node yaml = lifuren::yaml::loadFile(path);
    if(yaml.size() == 0LL) {
        return map;
    }
    std::for_each(yaml.begin(), yaml.end(), [&map](const auto& node) {
        const std::string& key = node.first.template as<std::string>();
        const auto& value      = node.second;
        Rhythm rhythm(key);
        LFR_RHYTHM_SETTER(value, rhythm,         rhythm, rhythm,         std::string);
        LFR_RHYTHM_SETTER(value, alias,          rhythm, alias,          std::vector<std::string>);
        LFR_RHYTHM_SETTER(value, title,          rhythm, title,          std::string);
        LFR_RHYTHM_SETTER(value, example,        rhythm, example,        std::string);
        LFR_RHYTHM_SETTER(value, fontSize,       rhythm, fontSize,       int);
        LFR_RHYTHM_SETTER(value, segmentSize,    rhythm, segmentSize,    int);
        LFR_RHYTHM_SETTER(value, segmentRule,    rhythm, segmentRule,    std::vector<uint32_t>);
        LFR_RHYTHM_SETTER(value, participleRule, rhythm, participleRule, std::vector<uint32_t>);
        map.emplace(key, rhythm);
    });
    return map;
}

std::set<std::string> lifuren::config::all_rhythm() {
    std::set<std::string> set;
    const auto& rhythm = lifuren::config::RHYTHM;
    for(const auto& [k, v] : rhythm) {
        set.emplace(k + " - " + v.title);
    }
    return set;
}

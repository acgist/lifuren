#include "lifuren/Config.hpp"

#include <sstream>

#include "spdlog/spdlog.h"

#include "yaml-cpp/yaml.h"

#include "lifuren/Yaml.hpp"

#ifndef LFR_RHYTHM_SETTER
#define LFR_RHYTHM_SETTER(source, key, target, field, type) \
const auto& field##Node = source[#key];                     \
if(field##Node) {                                           \
    target.field = field##Node.template as<type>();         \
}
#endif

std::map<std::string, lifuren::config::Rhythm> lifuren::config::RHYTHM{};

lifuren::config::Rhythm::Rhythm(const std::string& rhythm) : rhythm(rhythm) {
}

std::string lifuren::config::Rhythm::toYaml() {
    YAML::Node yaml;
    yaml["rhythm"]          = this->rhythm;
    yaml["alias"]           = this->alias;
    yaml["title"]           = this->title;
    yaml["example"]         = this->example;
    yaml["font-size"]       = this->fontSize;
    yaml["segment-size"]    = this->segmentSize;
    yaml["segment-rule"]    = this->segmentRule;
    yaml["participle-rule"] = this->participleRule;
    std::stringstream stream;
    stream << yaml;
    return stream.str();
}

std::map<std::string, lifuren::config::Rhythm> lifuren::config::Rhythm::loadFile() {
    const std::string path = lifuren::config::baseFile(lifuren::config::RHYTHM_PATH);
    SPDLOG_DEBUG("加载格律文件：{}", path);
    std::map<std::string, Rhythm> map;
    YAML::Node yaml = lifuren::yaml::loadFile(path);
    if(yaml.size() == 0) {
        return map;
    }
    std::for_each(yaml.begin(), yaml.end(), [&map](const auto& node) {
        const auto& key   = node.first.template as<std::string>();
        const auto& value = node.second;
        Rhythm rhythm(key);
        LFR_RHYTHM_SETTER(value, rhythm,          rhythm, rhythm,         std::string);
        LFR_RHYTHM_SETTER(value, alias,           rhythm, alias,          std::vector<std::string>);
        LFR_RHYTHM_SETTER(value, title,           rhythm, title,          std::string);
        LFR_RHYTHM_SETTER(value, example,         rhythm, example,        std::string);
        LFR_RHYTHM_SETTER(value, font-size,       rhythm, fontSize,       int);
        LFR_RHYTHM_SETTER(value, segment-size,    rhythm, segmentSize,    int);
        LFR_RHYTHM_SETTER(value, segment-rule,    rhythm, segmentRule,    std::vector<uint32_t>);
        LFR_RHYTHM_SETTER(value, participle-rule, rhythm, participleRule, std::vector<uint32_t>);
        map.emplace(key, std::move(rhythm));
    });
    return map;
}

std::set<std::string> lifuren::config::all_rhythm() {
    std::set<std::string> set;
    for(const auto& [k, v] : lifuren::config::RHYTHM) {
        set.emplace(k + " - " + v.title);
    }
    return set;
}

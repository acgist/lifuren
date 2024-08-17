#include "lifuren/config/Label.hpp"

#include <sstream>

#include "spdlog/spdlog.h"

#include "lifuren/Yamls.hpp"

lifuren::LabelFile::LabelFile() {
}

// TODO: move
lifuren::LabelFile::LabelFile(const std::string& name) : Label(name, name) {
}

std::string lifuren::LabelFile::toYaml() {
    YAML::Node yaml;
    yaml.push_back(this->labels);
    std::stringstream stream;
    stream << yaml;
    return stream.str();
}

std::map<std::string, std::vector<lifuren::LabelFile>> lifuren::LabelFile::loadFile(const std::string& path) {
    SPDLOG_DEBUG("加载标签文件：{}", path);
    std::map<std::string, std::vector<LabelFile>> map;
    YAML::Node yaml = lifuren::yamls::loadFile(path);
    if(yaml.size() == 0L) {
        return map;
    }
    for(auto iterator = yaml.begin(); iterator != yaml.end(); ++iterator) {
        const std::string key = iterator->first.as<std::string>();
        const auto value      = iterator->second;
        std::vector<LabelFile> vector;
        vector.reserve(value.size());
        for(auto labelIterator = value.begin(); labelIterator != value.end(); ++labelIterator) {
            LabelFile label(labelIterator->first.as<std::string>());
            // TODO: 减少拷贝
            label.labels = labelIterator->second.as<std::vector<std::string>>();
            vector.push_back(label);
        }
        map.emplace(key, std::move(vector));
    }
    return map;
}

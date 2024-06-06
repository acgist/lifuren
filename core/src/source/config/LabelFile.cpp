#include "../../header/config/Label.hpp"

#include "spdlog/spdlog.h"

lifuren::LabelFile::LabelFile() {
}

lifuren::LabelFile::LabelFile(const std::string& name, const YAML::Node& yaml) {
    this->name  = name;
    this->alias = name;
    this->labels = yaml.as<std::vector<std::string>>();
}

YAML::Node lifuren::LabelFile::toYaml() {
    YAML::Node yaml;
    yaml.push_back(this->labels);
    return yaml;
}

std::map<std::string, std::vector<lifuren::LabelFile>> lifuren::LabelFile::loadFile(const std::string& path) {
    SPDLOG_DEBUG("加载标签文件：{}", path);
    std::map<std::string, std::vector<LabelFile>> map;
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
        auto& value     = iterator->second;
        std::vector<LabelFile> vector;
        vector.reserve(value.size());
        for(
            auto labelIterator = value.begin();
            labelIterator != value.end();
            ++labelIterator
        ) {
            LabelFile label(
                labelIterator->first.as<std::string>(),
                labelIterator->second
            );
            vector.push_back(label);
        }
        map.emplace(key, std::move(vector));
    }
    return map;
    // 没有必要使用move
    // return std::move(map);
}

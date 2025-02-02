#include "lifuren/Yaml.hpp"

#include <fstream>

#include "spdlog/spdlog.h"

#include "yaml-cpp/yaml.h"

#include "lifuren/File.hpp"

YAML::Node lifuren::yaml::loadFile(const std::string& path) {
    if(!lifuren::file::exists(path) || !lifuren::file::is_file(path)) {
        return {};
    }
    return YAML::LoadFile(path);
}

bool lifuren::yaml::saveFile(const YAML::Node& yaml, const std::string& path) {
    lifuren::file::createParent(path);
    std::ofstream output;
    output.open(path, std::ios_base::out | std::ios_base::trunc);
    if(!output.is_open()) {
        SPDLOG_WARN("打开文件失败：{}", path);
        output.close();
        return false;
    }
    output << yaml;
    output.close();
    return true;
}

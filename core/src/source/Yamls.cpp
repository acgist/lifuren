#include "lifuren/Yamls.hpp"

#include <locale>
#include <fstream>

#include "spdlog/spdlog.h"

#include "yaml-cpp/yaml.h"

#include "lifuren/Files.hpp"
#include "lifuren/Logger.hpp"

YAML::Node lifuren::yamls::loadFile(const std::string& path) {
    if(!lifuren::files::exists(path) || !lifuren::files::isFile(path)) {
        return YAML::Node();
    }
    return YAML::LoadFile(path);
}

bool lifuren::yamls::saveFile(const YAML::Node& yaml, const std::string& path) {
    // 保存文件
    std::ofstream output;
    output.open(path, std::ios_base::out | std::ios_base::trunc);
    if(!output.is_open()) {
        SPDLOG_WARN("配置打开文件失败：{}", path);
        output.close();
        return false;
    }
    output << yaml;
    output.close();
    return true;
}

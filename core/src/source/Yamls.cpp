#include "lifuren/Yamls.hpp"

#include <locale>
#include <fstream>
#include <filesystem>

#include "lifuren/Logger.hpp"

#include "spdlog/spdlog.h"

YAML::Node lifuren::yamls::loadFile(const std::string& path) {
    if(!std::filesystem::exists(path) || !std::filesystem::is_regular_file(path)) {
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

/**
 * JSON工具
 * 
 * @author acgist
 */
#pragma once

#include <map>
#include <string>

#include "Files.hpp"
#include "nlohmann/json.hpp"

namespace lifuren {
namespace jsons {

template <typename T>
T loadFile(const std::string& path) {
    std::string json = lifuren::files::loadFile(path);
    if(json.empty()) {
        return {};
    }
    SPDLOG_DEBUG("加载JSON文件：{}", path);
    return nlohmann::json::parse(json);
}

extern void saveFile(const std::string& path, const nlohmann::json& json);

}
}
/**
 * JSON工具
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CORE_JSONS_HPP
#define LFR_HEADER_CORE_JSONS_HPP

#include <string>

#include "lifuren/Files.hpp"

#include "nlohmann/json.hpp"

namespace lifuren {
namespace jsons   {

/**
 * 加载JSON文件
 * 
 * @param path JSON文件路径
 * 
 * @return JSON内容
 */
template<typename T>
inline T loadFile(const std::string& path) {
    std::string json = lifuren::files::loadFile(path);
    if(json.empty()) {
        return {};
    }
    return nlohmann::json::parse(json);
}

/**
 * 保存JSON文件
 * 
 * @param json JSON内容
 * 
 * @return 是否成功
 */
extern bool saveFile(const std::string& path, const nlohmann::json& json);

} // END OF json
} // END OF lifuren

#endif // LFR_HEADER_CORE_JSONS_HPP

#include "../../header/utils/Jsons.hpp"

bool lifuren::jsons::saveFile(const std::string& path, const nlohmann::json& json) {
    SPDLOG_DEBUG("保存JSON文件：{}", path);
    return lifuren::files::saveFile(path, json.dump());
}

#include "../header/Jsons.hpp"

void lifuren::jsons::saveFile(const std::string& path, const nlohmann::json& json) {
    SPDLOG_DEBUG("保存JSON文件：{}", path);
    lifuren::files::saveFile(path, json.dump());
}

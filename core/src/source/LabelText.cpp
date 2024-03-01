#include "../header/Label.hpp"

std::string lifuren::LabelText::toJSON() {
    const nlohmann::json json = *this;
    return json.dump();
}

std::map<std::string, std::vector<lifuren::LabelFile>> lifuren::LabelFile::loadFile(const std::string& path) {
    std::map<std::string, std::vector<LabelFile>> map;
    const std::string text = lifuren::files::loadFile(path);
    if(text.empty()) {
        return map;
    }
    SPDLOG_DEBUG("加载标签文件：{}", path);
    const nlohmann::json json = nlohmann::json::parse(text);
    for(const auto& iter : json.items()) {
        std::string key      = iter.key();
        nlohmann::json value = iter.value();
        std::vector<LabelFile> vector;
        for(const auto& child : value.items()) {
            LabelFile label;
            label.name  = child.key();
            label.alias = child.key();
            auto value  = child.value();
            if(value.type() != nlohmann::json::value_t::array) {
                SPDLOG_WARN("不支持的标签类型：{} - {}", child.key(), value.type_name());
            } else {
                label.labels = child.value();
            }
            vector.push_back(label);
        }
        map.insert(std::pair(key, vector));
    }
    return map;
}


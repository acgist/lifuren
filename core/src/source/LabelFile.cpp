#include "../header/Label.hpp"

std::string lifuren::LabelFile::toJSON() {
    const nlohmann::json json = *this;
    return json.dump();
}

std::map<std::string, lifuren::LabelText> lifuren::LabelText::loadFile(const std::string& path) {
    std::map<std::string, LabelText> map;
    const std::string text = lifuren::files::loadFile(path);
    if(text.empty()) {
        return map;
    }
    SPDLOG_DEBUG("加载标签文件：{}", path);
    const nlohmann::json json = nlohmann::json::parse(text);
    for(const auto& iter : json.items()) {
        std::string key = iter.key();
        LabelText label = iter.value();
        if(label.name.empty()) {
            label.name = key;
        }
        map.insert(std::pair(key, label));
    }
    return map;
}

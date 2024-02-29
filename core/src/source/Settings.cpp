#include "../header/Setting.hpp"

lifuren::Settings lifuren::SETTINGS;

lifuren::Settings::Settings() {
}

lifuren::Settings::~Settings() {
}

void lifuren::Settings::load(const std::string& settings) {
    try {
        // 所有模型配置
        const char* keys[] = {
            "ImageGC",  "ImageTS",
            "AudioGC",  "AudioTS",
            "VideoGC",  "VideoTS",
            "PoetryGC", "PoetryTS",
        };
        const nlohmann::json json = nlohmann::json::parse(settings);
        for(const char* key : keys) {
            if(!json.contains(key)) {
                SPDLOG_DEBUG("缺少模型配置：{}", key);
                continue;
            }
            this->settings.insert(std::pair<std::string, lifuren::Setting>(key, json.at(key)));
        }
    } catch(const std::exception& e) {
        SPDLOG_ERROR("加载配置异常：{}", e.what());
    }
}

void lifuren::Settings::loadFile(const std::string& path) {
    std::string settings = lifuren::files::loadFile(path);
    if(settings.empty()) {
        return;
    }
    SPDLOG_DEBUG("加载配置文件：{}", path);
    SPDLOG_DEBUG("加载配置内容：{}", settings);
    *this = nlohmann::json::parse(settings);
}

void lifuren::Settings::saveFile(const std::string& path) {
    const std::string settings = this->toJSON();
    SPDLOG_DEBUG("保存配置路径：{}", path);
    SPDLOG_DEBUG("保存配置内容：{}", settings);
    lifuren::files::saveFile(path, settings);
}

std::string lifuren::Settings::toJSON() {
    const nlohmann::json json = *this;
    return json.dump();
}

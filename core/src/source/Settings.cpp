#include "../header/Setting.hpp"

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
                LOG(INFO) << "缺少模型配置：" << key;
                continue;
            }
            this->settings.insert(std::pair<std::string, lifuren::Setting>(key, json.at(key)));
        }
    } catch(const std::exception& e) {
        LOG(ERROR) << "加载配置异常：" << e.what();
    }
}

void lifuren::Settings::loadFile(const std::string& path) {
    std::ifstream input;
    input.open(path, std::ios_base::in);
    if(!input.is_open()) {
        LOG(WARNING) << "打开文件失败：" << path;
        return;
    }
    std::string line;
    std::string settings;
    while(std::getline(input, line)) {
        settings += line;
    }
    input.close();
    LOG(INFO) << "加载配置文件：" << path;
    LOG(INFO) << "加载配置内容：" << settings;
    *this = nlohmann::json::parse(settings);
}

void lifuren::Settings::saveFile(const std::string& path) {
    std::ofstream output;
    output.open(path, std::ios_base::out | std::ios_base::trunc);
    if(!output.is_open()) {
        LOG(WARNING) << "打开文件失败：" << path;
        return;
    }
    const std::string settings = this->toJSON();
    LOG(INFO) << "保存配置路径：" << path;
    LOG(INFO) << "保存配置内容：" << settings;
    output << settings;
    output.close();
}

std::string lifuren::Settings::toJSON() {
    const nlohmann::json json = *this;
    return json.dump();
}

#include "../header/Setting.hpp"

lifuren::Setting::Setting() {
    this->activation = lifuren::Activation::RELU;
    this->learningRate = 0.0;
    this->regularization = lifuren::Regularization::NONE;
    this->regularizationRate = 0.0;
}

lifuren::Setting::~Setting() {
}

lifuren::Setting::Setting(const std::string& json) {
    *this = nlohmann::json::parse(json);
}

std::string lifuren::Setting::toJSON() {
    const nlohmann::json json = *this;
    return json.dump();
}

void lifuren::Settings::load(const std::string& settings) {
    try {
        const nlohmann::json json = nlohmann::json::parse(settings);
        const char* keys[] = {
            "ImageGC",  "ImageTS",
            "AudioGC",  "AudioTS",
            "VideoGC",  "VideoTS",
            "PoetryGC", "PoetryTS",
        };
        for(const char* key : keys) {
            if(!json.contains(key)) {
                continue;
            }
            this->settings.insert(std::pair<std::string, lifuren::Setting>(key, json.at(key)));
        }
    } catch(const std::exception& e) {
        LOG(ERROR) << e.what();
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
    LOG(INFO) << "加载配置：" << settings;
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
    LOG(INFO) << "保存配置：" << settings;
    output << settings;
    output.close();
}

std::string lifuren::Settings::toJSON() {
    const nlohmann::json json = *this;
    return json.dump();
}

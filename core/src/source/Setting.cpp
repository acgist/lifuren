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
};

std::string lifuren::Setting::toJSON() {
    const nlohmann::json json = *this;
    return json.dump();
}

void lifuren::Settings::load(const std::string& settings) {
    try {
        nlohmann::json json = nlohmann::json::parse(settings);
        const char* keys[] = {
            "ImageGC", "ImageTS",
            "AudioGC", "AudioTS",
            "VideoGC", "VideoTS",
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

std::string lifuren::Settings::toJSON() {
    const nlohmann::json json = *this;
    return json.dump();
}

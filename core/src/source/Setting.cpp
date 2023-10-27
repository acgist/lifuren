#include "../header/Setting.hpp"

std::string lifuren::Setting::toJSON() {
    const nlohmann::json json = *this;
    return json.dump();
}

void lifuren::Settings::load(const std::string& settings) {
    try {
        nlohmann::json json = nlohmann::json::parse(settings);
        for(const char* key : { "ImageGC", "ImageTS", "VideoGC", "VideoTS", "PoetryGC" }) {
            if(!json.contains(key)) {
                continue;
            }
            this->settings.insert(std::pair<std::string, lifuren::Setting>(key, json.at(key)));
        }
    } catch(const std::exception& e) {
        LOG(ERROR) << e.what();
    }
}

#include "../../header/config/Setting.hpp"

#include "../header/utils/Jsons.hpp"

std::map<std::string, lifuren::Setting> lifuren::SETTINGS = lifuren::jsons::loadFile<std::map<std::string, lifuren::Setting>>(lifuren::SETTINGS_PATH);

lifuren::Setting::Setting() {
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

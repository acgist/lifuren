#include "../header/Setting.hpp"

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

#include "../header/Label.hpp"

lifuren::LabelConfig::LabelConfig() {
}

lifuren::LabelConfig::~LabelConfig() {
}

lifuren::LabelConfig::LabelConfig(const std::string& json) {
    *this = nlohmann::json::parse(json);
}

std::string lifuren::LabelConfig::toJSON() {
    const nlohmann::json json = *this;
    return json.dump();
}

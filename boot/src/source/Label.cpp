#include "../header/Label.hpp"

std::string lifuren::Label::toJSON() {
    const nlohmann::json json = *this;
    return json.dump();
}

std::string lifuren::LabelConfig::toJSON() {
    const nlohmann::json json = *this;
    return json.dump();
}

std::string lifuren::LabelSegment::toJSON() {
    const nlohmann::json json = *this;
    return json.dump();
}

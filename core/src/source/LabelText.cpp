#include "../header/Label.hpp"

lifuren::LabelText::LabelText() {
}

lifuren::LabelText::~LabelText() {
}

lifuren::LabelText::LabelText(const std::string& json) {
    *this = nlohmann::json::parse(json);
}

std::string lifuren::LabelText::toJSON() {
    const nlohmann::json json = *this;
    return json.dump();
}

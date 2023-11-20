#include "../header/Label.hpp"

lifuren::Label::Label() {
}

lifuren::Label::~Label() {
}

lifuren::Label::Label(const std::string& json) {
    *this = nlohmann::json::parse(json);
}

std::string lifuren::Label::toJSON() {
    const nlohmann::json json = *this;
    return json.dump();
}

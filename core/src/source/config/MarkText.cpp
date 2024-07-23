#include "lifuren/config/Mark.hpp"

lifuren::MarkText::MarkText() {
}

lifuren::MarkText::~MarkText() {
}

lifuren::MarkText::MarkText(const std::string& json) {
    *this = nlohmann::json::parse(json);
}

std::string lifuren::MarkText::toJSON() {
    const nlohmann::json json = *this;
    return json.dump();
}

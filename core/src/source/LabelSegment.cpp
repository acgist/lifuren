#include "../header/Label.hpp"

lifuren::LabelSegment::LabelSegment() {
}

lifuren::LabelSegment::~LabelSegment() {
}

lifuren::LabelSegment::LabelSegment(const std::string& json) {
    *this = nlohmann::json::parse(json);
}

std::string lifuren::LabelSegment::toJSON() {
    const nlohmann::json json = *this;
    return json.dump();
}

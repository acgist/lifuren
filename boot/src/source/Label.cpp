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

lifuren::LabelSegment::LabelSegment() {
    this->fontSize    = 0;
    this->segmentSize = 0;
}

lifuren::LabelSegment::~LabelSegment() {
    this->fontSize    = 0;
    this->segmentSize = 0;
}

lifuren::LabelSegment::LabelSegment(const std::string& json) {
    *this = nlohmann::json::parse(json);
}

std::string lifuren::LabelSegment::toJSON() {
    const nlohmann::json json = *this;
    return json.dump();
}

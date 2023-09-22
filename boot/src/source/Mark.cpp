#include "../header/Mark.hpp"

lifuren::Mark::Mark() {
}

lifuren::Mark::~Mark() {
}

lifuren::Mark::Mark(const std::string& json) {
    *this = nlohmann::json::parse(json);
}

std::string lifuren::Mark::toJSON() {
    const nlohmann::json json = *this;
    return json.dump();
}

lifuren::MarkFile::MarkFile() {
}

lifuren::MarkFile::~MarkFile() {
}

lifuren::MarkFile::MarkFile(const std::string& json) {
    *this = nlohmann::json::parse(json);
}

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
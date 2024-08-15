#include "lifuren/config/Mark.hpp"

lifuren::MarkFile::MarkFile() {
}

lifuren::MarkFile::~MarkFile() {
}

lifuren::MarkFile::MarkFile(const std::string& json) {
    *this = nlohmann::json::parse(json);
}

std::string lifuren::MarkFile::toJSON() {
    const nlohmann::json json = *this;
    return json.dump();
}

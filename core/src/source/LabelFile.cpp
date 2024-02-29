#include "../header/Label.hpp"

lifuren::LabelFile::LabelFile() {
}

lifuren::LabelFile::~LabelFile() {
}

lifuren::LabelFile::LabelFile(const std::string& json) {
    *this = nlohmann::json::parse(json);
}

std::string lifuren::LabelFile::toJSON() {
    const nlohmann::json json = *this;
    return json.dump();
}

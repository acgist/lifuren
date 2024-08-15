#include "lifuren/config/Mark.hpp"

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

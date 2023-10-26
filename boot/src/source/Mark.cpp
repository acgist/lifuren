#include "../header/Mark.hpp"

std::string lifuren::Mark::toJSON() {
    const nlohmann::json json = *this;
    return json.dump();
}

std::string lifuren::MarkText::toJSON() {
    const nlohmann::json json = *this;
    return json.dump();
}
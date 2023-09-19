#include "../header/Mark.hpp"

lifuren::Mark::Mark() {
}

lifuren::Mark::~Mark() {
}

std::string lifuren::Mark::toJSON() {
    const nlohmann::json json = *this;
    return json.dump();
}

lifuren::MarkFile::MarkFile() {
}

lifuren::MarkFile::~MarkFile() {
}

lifuren::MarkText::MarkText() {
}

lifuren::MarkText::~MarkText() {
}

std::string lifuren::MarkText::toJSON() {
    const nlohmann::json json = *this;
    return json.dump();
}
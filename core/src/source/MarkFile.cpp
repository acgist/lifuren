#include "../header/Mark.hpp"

lifuren::MarkFile::MarkFile() {
}

lifuren::MarkFile::~MarkFile() {
}

lifuren::MarkFile::MarkFile(const std::string& json) {
    *this = nlohmann::json::parse(json);
}

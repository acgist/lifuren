#include "../header/Strings.hpp"

#include <cctype>

void lifuren::strings::toLower(std::string& value) {
    // std::transform(value.begin(), value.end(), value.begin(), ::tolower);
    std::transform(value.begin(), value.end(), value.begin(), [](const char& v) -> char {
        // Linux宏定义
        return std::tolower(v);
    });
}

void lifuren::strings::toUpper(std::string& value) {
    // std::transform(value.begin(), value.end(), value.begin(), ::toupper);
    std::transform(value.begin(), value.end(), value.begin(), [](const char& v) -> char {
        // Linux宏定义
        return std::toupper(v);
    });
}

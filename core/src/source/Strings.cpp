#include "../header/Strings.hpp"

void lifuren::strings::toLower(std::string& value) {
    std::transform(value.begin(), value.end(), value.begin(), std::tolower);
}

void lifuren::strings::toUpper(std::string& value) {
    std::transform(value.begin(), value.end(), value.begin(), std::toupper);
}

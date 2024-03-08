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

void lifuren::strings::replace(std::string& value, const std::string& oldValue, const std::string& newValue) {
    std::string::size_type index = 0;
    std::string::size_type oldValueLength = oldValue.length();
    std::string::size_type newValueLength = newValue.length();
    while(true) {
        index = value.find(oldValue, index);
        if(index == std::string::npos) {
            break;
        }
        value.replace(index, oldValueLength, newValue);
        index += newValueLength;
    }
}

void lifuren::strings::replace(std::string& value, const std::vector<std::string>& oldValue, const std::string& newValue) {
    for(
        auto iterator = oldValue.begin();
        iterator != oldValue.end();
        ++iterator
    ) {
        lifuren::strings::replace(value, *iterator, newValue);
    }
}

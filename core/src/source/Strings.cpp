#include "lifuren/Strings.hpp"

#include <cctype>
#include <cstdint>
#include <cstring>
#include <algorithm>

void lifuren::strings::toLower(std::string& value) {
    #if _WIN32
    std::transform(value.begin(), value.end(), value.begin(), ::tolower);
    #else
    std::transform(value.begin(), value.end(), value.begin(), [](const char& v) -> char {
        return std::tolower(v);
    });
    #endif
}

void lifuren::strings::toUpper(std::string& value) {
    #if _WIN32
    std::transform(value.begin(), value.end(), value.begin(), ::toupper);
    #else
    std::transform(value.begin(), value.end(), value.begin(), [](const char& v) -> char {
        return std::toupper(v);
    });
    #endif
}

std::string lifuren::strings::trim(const std::string& value) {
    std::size_t index = value.find_first_not_of(EMPTY_CHARS);
    if(index == std::string::npos) {
        return std::string();
    }
    std::size_t jndex = value.find_last_not_of(EMPTY_CHARS);
    return value.substr(index, jndex + 1 - index);
}

char* lifuren::strings::trim(char* value) {
    const int size = std::strlen(value);
    char* index = value;
    char* jndex = value + size - 1;
    int length = size;
    while(index >= value && std::strchr(EMPTY_CHARS, *index)) {
        ++index;
        --length;
    }
    while(*jndex != '\0' && std::strchr(EMPTY_CHARS, *jndex)) {
        *jndex = '\0';
        --jndex;
        --length;
    }
    if(index <= jndex) {
        std::memmove(value, index, length + 1);
    }
    return value;
}

size_t lifuren::strings::length(const char* value) {
    size_t index = 0;
    size_t jndex = 0;
    while (value[index]) {
        if ((value[index] & 0xC0) != 0x80) {
            ++jndex;
        };
        ++index;
    }
    return jndex;
}

std::string lifuren::strings::substr(const char* value, uint32_t& pos, const uint32_t& length) {
    std::string ret;
    uint32_t index = 0;
    while(value[pos]) {
        ret.push_back(value[pos]);
        if((value[pos] & 0xC0) != 0x80) {
            ++index;
        };
        ++pos;
        if((value[pos] & 0xC0) != 0x80) {
            if(index >= length) {
                break;
            }
        };
    }
    return ret;
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

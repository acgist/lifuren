#include "lifuren/String.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>

static const char* const EMPTY_CHARS = " \t\r\n"; // 空白字符

std::vector<std::string> lifuren::string::split(const std::string& content, const std::string& delim, bool retain, bool filter) {
    size_t pos   = 0;
    size_t index = 0;
    std::string substr;
    std::vector<std::string> vector;
    while(true) {
        pos = content.find(delim, index);
        if(pos == std::string::npos) {
            break;
        }
        substr = content.substr(index, retain ? (pos - index + delim.length()) : (pos - index));
        if(filter) {
            substr = lifuren::string::trim(substr);
            if(!substr.empty()) {
                vector.push_back(std::move(substr));
            }
        } else {
            vector.push_back(std::move(substr));
        }
        index = pos + delim.length();
    }
    if(index <= content.length()) {
        substr = content.substr(index, content.length() - index);
        if(filter) {
            substr = lifuren::string::trim(substr);
            if(!substr.empty()) {
                vector.push_back(std::move(substr));
            }
        } else {
            vector.push_back(std::move(substr));
        }
    }
    return vector;
}

std::vector<std::string> lifuren::string::split(const std::string& content, const std::vector<std::string>& multi, bool retain, bool filter) {
    size_t pos   = 0;
    size_t index = 0;
    std::string delim;
    std::string substr;
    std::vector<std::string> vector;
    while(true) {
        size_t min = std::string::npos;
        for(const auto& value : multi) {
            pos = content.find(value, index);
            if(pos != std::string::npos && pos < min) {
                min   = pos;
                delim = value;
            }
        }
        pos = min;
        if(pos == std::string::npos) {
            break;
        }
        substr = content.substr(index, retain ? (pos - index + delim.length()) : (pos - index));
        if(filter) {
            substr = lifuren::string::trim(substr);
            if(!substr.empty()) {
                vector.push_back(std::move(substr));
            }
        } else {
            vector.push_back(std::move(substr));
        }
        index = pos + delim.length();
    }
    if(index <= content.length()) {
        substr = content.substr(index, content.length() - index);
        if(filter) {
            substr = lifuren::string::trim(substr);
            if(!substr.empty()) {
                vector.push_back(std::move(substr));
            }
        } else {
            vector.push_back(std::move(substr));
        }
    }
    return vector;
}

char* lifuren::string::trim(char* value) {
    const int   size   = std::strlen(value);
          char* index  = value;
          char* jndex  = value + size - 1;
          int   length = size;
    while(length >= 0 && std::strchr(EMPTY_CHARS, *index)) {
        ++index;
        --length;
    }
    while(length >= 0 && std::strchr(EMPTY_CHARS, *jndex)) {
        *jndex = '\0';
        --jndex;
        --length;
    }
    if(index <= jndex) {
        std::memmove(value, index, length + 1);
    }
    return value;
}

std::string lifuren::string::trim(const std::string& value) {
    std::size_t index = value.find_first_not_of(EMPTY_CHARS);
    if(index == std::string::npos) {
        return {};
    }
    std::size_t jndex = value.find_last_not_of(EMPTY_CHARS);
    return value.substr(index, jndex + 1 - index);
}

size_t lifuren::string::length(const char* value) {
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

uint32_t lifuren::string::indexPos(const char* value, uint32_t& pos, const uint32_t& size) {
    uint32_t index = 0;
    if(index < size) {
        while(value[pos]) {
            if((value[pos] & 0xC0) != 0x80) {
                ++index;
            }
            ++pos;
            if((value[pos] & 0xC0) != 0x80) {
                if(index >= size) {
                    break;
                }
            }
        }
    }
    return pos;
}

std::string lifuren::string::substr(const char* value, const uint32_t& offset, const uint32_t& length) {
    std::string ret;
    uint32_t pos = 0;
    uint32_t beg = lifuren::string::indexPos(value, pos, offset);
    uint32_t end = lifuren::string::indexPos(value, pos, length);
    ret.insert(ret.begin(), value + beg, value + end);
    return ret;
}

std::vector<std::string> lifuren::string::toChars(const std::string& segment, bool filter) {
    std::string ret;
    uint32_t pos   = 0;
    uint32_t index = 0;
    std::vector<std::string> vector;
    const char* value = segment.c_str();
    while(value[pos]) {
        ret.push_back(value[pos]);
        if((value[pos] & 0xC0) != 0x80) {
            ++index;
        };
        ++pos;
        if((value[pos] & 0xC0) != 0x80) {
            if(filter) {
                ret = lifuren::string::trim(ret);
                if(!ret.empty()) {
                    vector.push_back(std::move(ret));
                }
            } else {
                vector.push_back(std::move(ret));
            }
        };
    }
    return vector;
}

void lifuren::string::replace(std::string& value, const std::string& oldValue, const std::string& newValue) {
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

void lifuren::string::replace(std::string& value, const std::vector<std::string>& oldValue, const std::string& newValue) {
    std::for_each(oldValue.begin(), oldValue.end(), [&value, &newValue](const auto& v) {
        lifuren::string::replace(value, v, newValue);
    });
}

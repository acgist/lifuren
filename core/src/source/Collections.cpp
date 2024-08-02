#include "lifuren/Collections.hpp"

std::vector<std::string> lifuren::collections::split(const std::string& content, const std::string& delim, bool retain, bool filter) {
    std::vector<std::string> vector;
    size_t pos   = 0;
    size_t index = 0;
    std::string substr;
    while(true) {
        pos = content.find(delim, index);
        if(pos == std::string::npos) {
            break;
        }
        substr = content.substr(index, retain ? pos - index + delim.length() : pos - index);
        if(!filter || !substr.empty()) {
            vector.push_back(substr);
        }
        index = pos + delim.length();
    }
    if(pos != index && index <= content.length()) {
        substr = content.substr(index, content.length() - index);
        if(!filter || !substr.empty()) {
            vector.push_back(substr);
        }
    }
    return vector;
}

std::vector<std::string> lifuren::collections::split(const std::string& content, const std::vector<std::string>& multi, bool retain, bool filter) {
    std::vector<std::string> vector;
    size_t pos   = 0;
    size_t index = 0;
    std::string delim;
    std::string substr;
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
        substr = content.substr(index, retain ? pos - index + delim.length() : pos - index);
        if(!filter || !substr.empty()) {
            vector.push_back(substr);
        }
        index = pos + delim.length();
    }
    if(pos != index && index <= content.length()) {
        substr = content.substr(index, content.length() - index);
        if(!filter || !substr.empty()) {
            vector.push_back(substr);
        }
    }
    return vector;
}

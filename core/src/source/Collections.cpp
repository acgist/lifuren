#include "Collections.hpp"

std::vector<std::string> lifuren::collections::split(const std::string& content, const std::string& delim) {
    return lifuren::collections::split(content, delim, false);
}

std::vector<std::string> lifuren::collections::split(const std::string& content, const std::string& delim, bool retain) {
    std::vector<std::string> vector;
    size_t pos   = 0;
    size_t index = 0;
    while(true) {
        pos = content.find(delim, index);
        if(pos == std::string::npos) {
            break;
        }
        vector.push_back(content.substr(index, retain ? pos - index + delim.length() : pos - index));
        index = pos + delim.length();
    }
    if(pos != index && index <= content.length()) {
        vector.push_back(content.substr(index, content.length() - index));
    }
    return vector;
}

std::vector<std::string> lifuren::collections::split(const std::string& content, const std::vector<std::string>& multi) {
    return lifuren::collections::split(content, multi, false);
}

std::vector<std::string> lifuren::collections::split(const std::string& content, const std::vector<std::string>& multi, bool retain) {
    std::vector<std::string> vector;
    size_t pos   = 0;
    size_t index = 0;
    std::string delim;
    while(true) {
        size_t min = std::string::npos;
        for(auto& value : multi) {
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
        vector.push_back(content.substr(index, retain ? pos - index + delim.length() : pos - index));
        index = pos + delim.length();
    }
    if(pos != index && index <= content.length()) {
        vector.push_back(content.substr(index, content.length() - index));
    }
    return vector;
}

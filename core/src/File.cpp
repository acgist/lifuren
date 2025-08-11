#include "lifuren/File.hpp"

#include "spdlog/spdlog.h"

/**
 * @param value 字符串
 */
inline void to_lower(std::string& value) {
    #if _WIN32
    std::transform(value.begin(), value.end(), value.begin(), ::tolower);
    #else
    std::transform(value.begin(), value.end(), value.begin(), [](const char& v) -> char {
        return std::tolower(v);
    });
    #endif
}

void lifuren::file::list_file(std::vector<std::string>& vector, const std::string& path) {
    lifuren::file::list_file(vector, path, [](const std::string&) {
        return true;
    });
}

void lifuren::file::list_file(std::vector<std::string>& vector, const std::string& path, const std::vector<std::string>& suffix) {
    lifuren::file::list_file(vector, path, [&suffix](const std::string& filename) -> bool {
        if(suffix.empty()) {
            return true;
        } else {
            const size_t pos = filename.find_last_of('.');
            if(pos == std::string::npos) {
                return false;
            }
            std::string file_suffix = filename.substr(pos);
            ::to_lower(file_suffix);
            return std::find(suffix.begin(), suffix.end(), file_suffix) != suffix.end();
        }
    });
}

void lifuren::file::list_file(std::vector<std::string>& vector, const std::string& path, const std::function<bool(const std::string&)>& predicate) {
    if(!lifuren::file::is_directory(path)) {
        SPDLOG_DEBUG("无效目录：{}", path);
        return;
    }
    const auto iterator = std::filesystem::directory_iterator(std::filesystem::path(path));
    for(const auto& entry : iterator) {
        std::string filepath = entry.path().string();
        if(entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            if(predicate(filename)) {
                vector.push_back(filepath);
            } else {
                SPDLOG_DEBUG("无效文件：{}", filepath);
            }
        } else {
            SPDLOG_DEBUG("忽略文件：{}", filepath);
        }
    }
}

#include "lifuren/File.hpp"

#include "spdlog/spdlog.h"

void lifuren::file::list_file(std::vector<std::string>& vector, const std::string& path) {
    lifuren::file::list_file(vector, path, [](const std::string&) {
        return true;
    });
}

void lifuren::file::list_file(std::vector<std::string>& vector, const std::string& path, const std::vector<std::string>& suffix) {
    lifuren::file::list_file(vector, path, [&vector, &suffix](const std::string& filename) -> bool {
        if(suffix.size() == 0) {
            return true;
        } else {
            const size_t pos = filename.find_last_of('.');
            if(pos == std::string::npos) {
                return false;
            }
            std::string file_suffix = filename.substr(pos);
            lifuren::string::toLower(file_suffix);
            const auto ret = std::find(suffix.begin(), suffix.end(), file_suffix);
            return ret != suffix.end();
        }
    });
}

void lifuren::file::list_file(std::vector<std::string>& vector, const std::string& path, const std::function<bool(const std::string&)>& predicate) {
    if(!lifuren::file::exists(path) || !lifuren::file::is_directory(path)) {
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

std::string lifuren::file::file_suffix(const std::string& path) {
    if(path.empty()) {
        return {};
    }
    const auto pos = path.find_last_of('.');
    if(pos == path.npos) {
        return {};
    }
    auto suffix = path.substr(pos);
    lifuren::string::toLower(suffix);
    return suffix;
}

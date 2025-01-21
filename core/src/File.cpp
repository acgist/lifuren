#include "lifuren/File.hpp"

#include <fstream>
#include <algorithm>

#include "spdlog/spdlog.h"

#include "lifuren/String.hpp"

void lifuren::file::listFile(std::vector<std::string>& vector, const std::string& path) {
    listFile(vector, path, [](const std::string&) {
        return true;
    });
}

void lifuren::file::listFile(std::vector<std::string>& vector, const std::string& path, const std::vector<std::string>& suffix) {
    listFile(vector, path, [&](const std::string& filename) -> bool {
        if(suffix.empty()) {
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

void lifuren::file::listFile(std::vector<std::string>& vector, const std::string& path, const std::function<bool(const std::string&)>& predicate) {
    if(!exists(path) || !is_directory(path)) {
        SPDLOG_DEBUG("目录无效：{}", path);
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
                SPDLOG_DEBUG("忽略无效文件类型：{}", filepath);
            }
        } else {
            SPDLOG_DEBUG("忽略无效文件：{}", filepath);
        }
    }
}

std::string lifuren::file::loadFile(const std::string& path) {
    std::ifstream input;
    input.open(path, std::ios_base::in);
    if(!input.is_open()) {
        SPDLOG_WARN("打开文件失败：{}", path);
        input.close();
        return {};
    }
    std::string line;
    std::string lines;
    while(std::getline(input, line)) {
        lines += line + '\n';
    }
    input.close();
    return lines;
}

std::vector<char> lifuren::file::loadBlobFile(const std::string& path) {
    const size_t length = std::filesystem::file_size(std::filesystem::path(path));
    if(length == 0) {
        return {};
    }
    std::ifstream input;
    input.open(path, std::ios_base::in | std::ios_base::binary);
    if(!input.is_open()) {
        SPDLOG_WARN("打开文件失败：{}", path);
        input.close();
        return {};
    }
    std::vector<char> blob;
    blob.resize(length);
    input.read(blob.data(), length);
    input.close();
    return blob;
}

bool lifuren::file::saveFile(const std::string& path, const std::string& value) {
    lifuren::file::createParent(path);
    std::ofstream output;
    output.open(path, std::ios_base::out | std::ios_base::trunc);
    if(!output.is_open()) {
        SPDLOG_WARN("打开文件失败：{}", path);
        output.close();
        return false;
    }
    output << value;
    output.close();
    return true;
}

std::string lifuren::file::fileSuffix(const std::string& path) {
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

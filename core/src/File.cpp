#include "lifuren/File.hpp"

#include <fstream>
#include <algorithm>

#include "spdlog/spdlog.h"

#include "lifuren/String.hpp"

void lifuren::file::listFile(std::vector<std::string>& vector, const std::string& path) {
    listFile(vector, path, [](const std::string&) { return true; });
}

void lifuren::file::listFile(std::vector<std::string>& vector, const std::string& path, const std::vector<std::string>& exts) {
    listFile(vector, path, [&](const std::string& filename) -> bool {
        if(exts.empty()) {
            return true;
        } else {
            const size_t pos = filename.find_last_of('.');
            if(pos == std::string::npos) {
                return false;
            }
            std::string ext = filename.substr(pos);
            lifuren::string::toLower(ext);
            const auto ret = std::find(exts.begin(), exts.end(), ext);
            return ret != exts.end();
        }
    });
}

void lifuren::file::listFile(std::vector<std::string>& vector, const std::string& path, const std::function<bool(const std::string& path)>& predicate) {
    if(!exists(path) || !isDirectory(path)) {
        SPDLOG_DEBUG("目录无效：{}", path);
        return;
    }
    const auto iterator = std::filesystem::directory_iterator(std::filesystem::u8path(path));
    for(const auto& entry : iterator) {
        // TODO: utf8
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
        return "";
    }
    std::string line;
    std::string lines;
    while(std::getline(input, line)) {
        lines += line + '\n';
    }
    input.close();
    return lines;
}

void lifuren::file::loadFile(const std::string& path, char** data, size_t& length) {
    length = std::filesystem::file_size(std::filesystem::u8path(path));
    if(length == 0LL) {
        return;
    }
    std::ifstream input;
    input.open(path, std::ios_base::in | std::ios_base::binary);
    if(!input.is_open()) {
        SPDLOG_WARN("打开文件失败：{}", path);
        input.close();
        return;
    }
    *data = new char[length];
    input.read(*data, length);
    input.close();
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

bool lifuren::file::createParent(const std::string& path) {
    const std::filesystem::path file { std::filesystem::u8path(path) };
    const auto parent = file.parent_path();
    if(std::filesystem::exists(parent)) {
        return true;
    } else {
        return std::filesystem::create_directories(parent);
    }
}

bool lifuren::file::createFolder(const std::string& path) {
    const std::filesystem::path file { std::filesystem::u8path(path) };
    if(std::filesystem::exists(file)) {
        return true;
    } else {
        return std::filesystem::create_directories(file);
    }
}

std::string lifuren::file::fileType(const std::string& path) {
     const std::filesystem::path file { std::filesystem::u8path(path) };
     std::string extension = file.extension().string();
     if(extension.empty()) {
        return extension;
     }
     if(extension.at(0) == '.') {
        return extension.substr(1);
     }
     return extension;
}

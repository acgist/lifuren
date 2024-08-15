#include "lifuren/DocumentReader.hpp"

#include <filesystem>

#include "spdlog/spdlog.h"

lifuren::TextReader::TextReader(const std::string& path) : DocumentReader(path) {
    this->input.open(path, std::ios_base::in);
    if(!this->input.is_open()) {
        SPDLOG_WARN("打开文件失败：{}", path);
        this->input.close();
        return;
    }
    this->fileSize = std::filesystem::file_size(path);
}

lifuren::TextReader::~TextReader() {
    this->input.close();
}

std::string lifuren::TextReader::readAll() {
    std::string content;
    if(!this->input) {
        return content;
    }
    std::string line;
    while(std::getline(this->input, line)) {
        content += line + '\n';
    }
    return content;
}

bool lifuren::TextReader::hasMore() {
    return this->input && !this->input.eof();
}

std::string lifuren::TextReader::readMore() {
    std::string line;
    std::getline(this->input, line);
    line += '\n';
    return line;
}

float lifuren::TextReader::percent() {
    if(this->fileSize <= 0L) {
        return 1.0F;
    }
    const auto pos = this->input.tellg();
    if(pos < 0) {
        return 1.0F;
    }
    return static_cast<double>(pos) / this->fileSize;
}

bool lifuren::TextReader::reset() {
    this->input.clear();
    this->input.seekg(std::ios_base::beg);
    return true;
}

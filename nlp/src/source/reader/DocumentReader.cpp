#include "lifuren/DocumentReader.hpp"

#include <filesystem>

#include "spdlog/spdlog.h"

#include "lifuren/Files.hpp"
#include "lifuren/Strings.hpp"

lifuren::DocumentReader::DocumentReader(const std::string& path) : path(path) {
    if(!std::filesystem::exists(path)) {
        SPDLOG_INFO("文档文件无效：{}", path);
    }
}

lifuren::DocumentReader::~DocumentReader() {
}

std::unique_ptr<lifuren::DocumentReader> lifuren::DocumentReader::getReader(const std::string& path) {
    std::string&& fileType = lifuren::files::fileType(path);
    lifuren::strings::toLower(fileType);
    if(fileType == "md") {
        return std::make_unique<lifuren::MarkdownReader>(path);
    } else if(fileType == "txt") {
        return std::make_unique<lifuren::TextReader>(path);
    } else if(fileType == "pdf") {
        return std::make_unique<lifuren::PDFReader>(path);
    } else if(fileType == "docx") {
        return std::make_unique<lifuren::WordReader>(path);
    } else {
        SPDLOG_WARN("不支持的文档格式：{} - {}", fileType, path);
    }
    return nullptr;
}

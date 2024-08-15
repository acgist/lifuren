#include "lifuren/DocumentReader.hpp"

#include <filesystem>

#include "spdlog/spdlog.h"

lifuren::DocumentReader::DocumentReader(const std::string& path) : path(path) {
    if(!std::filesystem::exists(path)) {
        SPDLOG_INFO("文档文件无效：{}", path);
    }
}

lifuren::DocumentReader::~DocumentReader() {
}

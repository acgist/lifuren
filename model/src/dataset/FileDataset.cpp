#include "lifuren/Dataset.hpp"

#include <random>
#include <algorithm>

#include "spdlog/spdlog.h"

#include "lifuren/File.hpp"

lifuren::dataset::FileDataset::FileDataset(
    const std::string& path,
    const std::vector<std::string>& exts,
    const std::map<std::string, float>& classify,
    const std::function<torch::Tensor(const std::string&)> transform
) : transform(transform) {
    if(!lifuren::file::exists(path) || !lifuren::file::isDirectory(path)) {
        SPDLOG_DEBUG("目录无效：{}", path);
        return;
    }
    auto iterator = std::filesystem::directory_iterator(std::filesystem::u8path(path));
    for(const auto& entry : iterator) {
        const auto path = entry.path();
        if(entry.is_directory()) {
            lifuren::file::listFile(this->features, path.string(), exts);
            this->labels.resize(this->features.size(), torch::full({ 1 }, classify.at(path.filename().string()), torch::kFloat32));
        } else {
            SPDLOG_DEBUG("忽略无效文件：{}", path.string());
        }
    }
}

lifuren::dataset::FileDataset::FileDataset(
    const std::string& path,
    const std::vector<std::string>& exts,
    const std::function<torch::Tensor(const std::string&)> mapping,
    const std::function<torch::Tensor(const std::string&)> transform
) : transform(transform) {
    if(!lifuren::file::exists(path) || !lifuren::file::isDirectory(path)) {
        SPDLOG_DEBUG("目录无效：{}", path);
        return;
    }
    auto iterator = std::filesystem::directory_iterator(std::filesystem::u8path(path));
    for(const auto& entry : iterator) {
        const auto path = entry.path();
        if(entry.is_directory()) {
            lifuren::file::listFile(this->features, path.string(), exts);
        } else {
            SPDLOG_DEBUG("忽略无效文件：{}", path.string());
        }
    }
    for(const auto& path : this->features) {
        this->labels.push_back(mapping(path));
    }
}

lifuren::dataset::FileDataset::~FileDataset() {
}

torch::optional<size_t> lifuren::dataset::FileDataset::size() const {
    return this->features.size();
}

torch::data::Example<> lifuren::dataset::FileDataset::get(size_t index) {
    torch::Tensor feature = this->transform(this->features.at(index));
    torch::Tensor label   = this->labels.at(index);
    return { 
        feature,
        label
    };
}

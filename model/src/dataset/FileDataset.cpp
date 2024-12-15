#include "lifuren/Dataset.hpp"

#include <fstream>
#include <algorithm>

#include "spdlog/spdlog.h"

#include "lifuren/File.hpp"
#include "lifuren/Torch.hpp"
#include "lifuren/Config.hpp"

lifuren::dataset::FileDataset::FileDataset(
    const std::string& path,
    const std::vector<std::string>& exts,
    const std::map<std::string, float>& classify,
    const std::function<torch::Tensor(const std::string&, const torch::DeviceType&)> transform
) {
    if(!lifuren::file::exists(path) || !lifuren::file::isDirectory(path)) {
        SPDLOG_DEBUG("目录无效：{}", path);
        return;
    }
    lifuren::setDevice(this->device);
    auto iterator = std::filesystem::directory_iterator(std::filesystem::u8path(path));
    for(const auto& entry : iterator) {
        const auto path = entry.path();
        if(entry.is_directory() && path.string() != lifuren::config::LIFUREN_HIDDEN_FILE) {
            std::vector<std::string> files;
            lifuren::file::listFile(files, path.string(), exts);
            for(const auto& file : files) {
                SPDLOG_DEBUG("加载文件：{}", file);
                this->features.push_back(std::move(transform(file, this->device)));
            }
            this->labels.resize(this->features.size(), torch::full({ 1 }, classify.at(path.filename().string()), torch::kFloat32).to(this->device));
        } else {
            SPDLOG_DEBUG("忽略无效文件：{}", path.string());
        }
    }
}

lifuren::dataset::FileDataset::FileDataset(
    const std::string& path,
    const std::function<void(const std::string&, std::vector<torch::Tensor>&, std::vector<torch::Tensor>&, const torch::DeviceType&)> transform
) {
    if(!lifuren::file::exists(path) || !lifuren::file::isFile(path)) {
        SPDLOG_DEBUG("文件无效：{}", path);
        return;
    }
    lifuren::setDevice(this->device);
    SPDLOG_DEBUG("加载文件：{}", path);
    transform(path, this->labels, this->features, this->device);
}

lifuren::dataset::FileDataset::FileDataset(
    const std::string& path,
    const std::vector<std::string>& exts,
    const std::function<void(const std::string&, std::vector<torch::Tensor>&, std::vector<torch::Tensor>&, const torch::DeviceType&)> transform
) {
    if(!lifuren::file::exists(path) || !lifuren::file::isDirectory(path)) {
        SPDLOG_DEBUG("目录无效：{}", path);
        return;
    }
    lifuren::setDevice(this->device);
    std::vector<std::string> files;
    lifuren::file::listFile(files, path, exts);
    for(const auto& file : files) {
        SPDLOG_DEBUG("加载文件：{}", file);
        transform(file, this->labels, this->features, this->device);
    }
}

lifuren::dataset::FileDataset::FileDataset(
    const std::string& path,
    const std::string& label,
    const std::vector<std::string>& exts,
    const std::function<void(const std::string&, const std::string&, std::vector<torch::Tensor>&, std::vector<torch::Tensor>&, const torch::DeviceType&)> transform
) {
    if(!lifuren::file::exists(path) || !lifuren::file::isDirectory(path)) {
        SPDLOG_DEBUG("目录无效：{}", path);
        return;
    }
    lifuren::setDevice(this->device);
    std::vector<std::string> files;
    lifuren::file::listFile(files, path, exts);
    for(const auto& file : files) {
        const auto index = file.find_last_of('.');
        if(index == std::string::npos) {
            SPDLOG_INFO("加载文件没有标记文件：{}", file);
            continue;
        }
        const auto label_file = file.substr(0, index) + label;
        if(!lifuren::file::exists(label_file)) {
            SPDLOG_INFO("加载文件没有标记文件：{}", file);
            continue;
        }
        SPDLOG_DEBUG("加载文件：{} - {}", file, label_file);
        transform(file, label_file, this->labels, this->features, this->device);
    }
}

lifuren::dataset::FileDataset::FileDataset(
    const std::string& path,
    const std::string& source,
    const std::string& target,
    const std::vector<std::string>& exts,
    const std::function<void(const std::string&, const std::string&, std::vector<torch::Tensor>&, std::vector<torch::Tensor>&, const torch::DeviceType&)> transform
) {
    if(!lifuren::file::exists(path) || !lifuren::file::isDirectory(path)) {
        SPDLOG_DEBUG("目录无效：{}", path);
        return;
    }
    lifuren::setDevice(this->device);
    std::vector<std::string> files;
    lifuren::file::listFile(files, path, exts);
    for(const auto& file : files) {
        const auto index = file.find_last_of('.');
        if(index == std::string::npos) {
            SPDLOG_INFO("加载文件匹配规则失败：{}", file);
            continue;
        }
        if(index < source.size()) {
            SPDLOG_INFO("加载文件匹配规则失败：{}", file);
            continue;
        }
        const auto label = file.substr(index - source.size(), source.size());
        if(label != source) {
            if(label != target) {
                SPDLOG_INFO("加载文件匹配规则失败：{}", file);
            }
            continue;
        }
        auto target_file(file);
        target_file.replace(index - source.size(), source.size(), target);
        const auto iterator = std::find(files.begin(), files.end(), target_file);
        if(iterator == files.end()) {
            SPDLOG_INFO("加载文件没有标记文件：{}", file);
            continue;
        }
        SPDLOG_DEBUG("加载文件：{} - {}", file, target_file);
        transform(file, target_file, this->labels, this->features, this->device);
    }
}

lifuren::dataset::FileDataset::~FileDataset() {
}

torch::optional<size_t> lifuren::dataset::FileDataset::size() const {
    return this->features.size();
}

torch::data::Example<> lifuren::dataset::FileDataset::get(size_t index) {
    return {
        this->features[index],
        this->labels  [index]
    };
}

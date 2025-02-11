#include "lifuren/Dataset.hpp"

#include <fstream>
#include <algorithm>

#include "spdlog/spdlog.h"

#include "lifuren/File.hpp"
#include "lifuren/Torch.hpp"
#include "lifuren/Config.hpp"

lifuren::dataset::FileDataset::FileDataset(
    const std::string                 & path,
    const std::vector<std::string>    & suffix,
    const std::map<std::string, float>& classify,
    const std::function<torch::Tensor(
        const std::string      &,
        const torch::DeviceType&
    )> transform
) : device(lifuren::getDevice()) {
    if(!lifuren::file::exists(path) || !lifuren::file::is_directory(path)) {
        SPDLOG_WARN("目录无效：{}", path);
        return;
    }
    auto iterator = std::filesystem::directory_iterator(std::filesystem::path(path));
    for(const auto& entry : iterator) {
        const auto entry_path = entry.path();
        if(entry.is_directory() && entry_path.string() != lifuren::config::LIFUREN_HIDDEN_FILE) {
            std::vector<std::string> files;
            lifuren::file::listFile(files, entry_path.string(), suffix);
            for(const auto& file : files) {
                SPDLOG_DEBUG("加载文件：{}", file);
                this->features.push_back(std::move(transform(file, this->device)));
            }
            this->labels.resize(this->features.size(), torch::full({ 1 }, classify.at(entry_path.filename().string()), torch::kFloat32).to(this->device));
        } else {
            SPDLOG_DEBUG("忽略文件：{}", entry_path.string());
        }
    }
}

lifuren::dataset::FileDataset::FileDataset(
    const std::string& path,
    const std::function<void(
        const std::string         &,
        std::vector<torch::Tensor>&,
        std::vector<torch::Tensor>&,
        const torch::DeviceType   &
    )> transform
) : device(lifuren::getDevice()) {
    if(!lifuren::file::exists(path) || !lifuren::file::is_file(path)) {
        SPDLOG_WARN("文件无效：{}", path);
        return;
    }
    SPDLOG_DEBUG("加载文件：{}", path);
    transform(path, this->labels, this->features, this->device);
}

lifuren::dataset::FileDataset::FileDataset(
    const std::string             & path,
    const std::vector<std::string>& suffix,
    const std::function<void(
        const std::string         &,
        std::vector<torch::Tensor>&,
        std::vector<torch::Tensor>&,
        const torch::DeviceType   &
    )> transform
) : device(lifuren::getDevice()) {
    if(!lifuren::file::exists(path) || !lifuren::file::is_directory(path)) {
        SPDLOG_WARN("目录无效：{}", path);
        return;
    }
    std::vector<std::string> files;
    lifuren::file::listFile(files, path, suffix);
    for(const auto& file : files) {
        SPDLOG_DEBUG("加载文件：{}", file);
        transform(file, this->labels, this->features, this->device);
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

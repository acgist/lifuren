#include "lifuren/Dataset.hpp"

#include "spdlog/spdlog.h"

#include "lifuren/File.hpp"

lifuren::dataset::FileDataset::FileDataset(
    std::vector<torch::Tensor>& labels,
    std::vector<torch::Tensor>& features
) : labels(std::move(labels)), features(std::move(features)) {
}

lifuren::dataset::FileDataset::FileDataset(
    const std::string& path,
    const std::vector<std::string>& exts,
    const std::map<std::string, float>& classify,
    const std::function<torch::Tensor(const std::string&)> transform
) {
    if(!lifuren::file::exists(path) || !lifuren::file::isDirectory(path)) {
        SPDLOG_DEBUG("目录无效：{}", path);
        return;
    }
    auto iterator = std::filesystem::directory_iterator(std::filesystem::u8path(path));
    for(const auto& entry : iterator) {
        const auto path = entry.path();
        if(entry.is_directory()) {
            std::vector<std::string> files;
            lifuren::file::listFile(files, path.string(), exts);
            for(const auto& file : files) {
                this->features.push_back(std::move(transform(file)));
            }
            this->labels.resize(this->features.size(), torch::full({ 1 }, classify.at(path.filename().string()), torch::kFloat32));
        } else {
            SPDLOG_DEBUG("忽略无效文件：{}", path.string());
        }
    }
}

lifuren::dataset::FileDataset::FileDataset(
    const std::string& path,
    const std::vector<std::string>& exts,
    const std::function<void(const std::ifstream&, std::vector<torch::Tensor>&, std::vector<torch::Tensor>&)> transform
) {
    if(!lifuren::file::exists(path) || !lifuren::file::isDirectory(path)) {
        SPDLOG_DEBUG("目录无效：{}", path);
        return;
    }
    std::vector<std::string> files;
    lifuren::file::listFile(files, path, exts);
    for(const auto& file : files) {
        std::ifstream stream;
        stream.open(file, std::ios_base::in | std::ios_base::binary);
        if(stream.is_open()) {
            transform(stream, this->labels, this->features);
        } else {
            SPDLOG_WARN("文件打开失败：{}", file);
        }
        stream.close();
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
        this->labels[index]
    };
}

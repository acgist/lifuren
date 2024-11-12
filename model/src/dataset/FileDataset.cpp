#include "lifuren/Dataset.hpp"

#include "spdlog/spdlog.h"

#include "lifuren/File.hpp"
#include "lifuren/Torch.hpp"

lifuren::dataset::FileDataset::FileDataset(
    std::vector<torch::Tensor>& labels,
    std::vector<torch::Tensor>& features
) : labels(std::move(labels)), features(std::move(features)) {
    lifuren::setDevice(this->device);
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
    lifuren::setDevice(this->device);
    auto iterator = std::filesystem::directory_iterator(std::filesystem::u8path(path));
    for(const auto& entry : iterator) {
        const auto path = entry.path();
        if(entry.is_directory()) {
            std::vector<std::string> files;
            lifuren::file::listFile(files, path.string(), exts);
            for(const auto& file : files) {
                this->features.push_back(std::move(transform(file).to(this->device)));
            }
            this->labels.resize(this->features.size(), torch::full({ 1 }, classify.at(path.filename().string()), torch::kFloat32).to(this->device));
        } else {
            SPDLOG_DEBUG("忽略无效文件：{}", path.string());
        }
    }
}

lifuren::dataset::FileDataset::FileDataset(
    const std::string& path,
    const std::function<void(std::ifstream&, std::vector<torch::Tensor>&, std::vector<torch::Tensor>&, const torch::DeviceType&)> transform
) {
    if(!lifuren::file::exists(path) || !lifuren::file::isFile(path)) {
        SPDLOG_DEBUG("目录无效：{}", path);
        return;
    }
    lifuren::setDevice(this->device);
    std::ifstream stream;
    stream.open(path, std::ios_base::in | std::ios_base::binary);
    if(stream.is_open()) {
        transform(stream, this->labels, this->features, this->device);
    } else {
        SPDLOG_WARN("文件打开失败：{}", path);
    }
    stream.close();
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

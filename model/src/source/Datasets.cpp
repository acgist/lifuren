#include "lifuren/Datasets.hpp"

#include "lifuren/Logger.hpp"
#include "lifuren/Files.hpp"

#include "spdlog/spdlog.h"

lifuren::datasets::FileDataset::FileDataset(
    const std::string& path,
    const std::vector<std::string>& exts,
    const std::map<std::string, int>& mapping,
    const std::function<torch::Tensor(const std::string&)> fileTransform
) : fileTransform(fileTransform) {
    if(!std::filesystem::exists(path) || !std::filesystem::is_directory(path)) {
        SPDLOG_DEBUG("目录无效：{}", path);
        return;
    }
    auto iterator = std::filesystem::directory_iterator(std::filesystem::path(path));
    for(const auto& entry : iterator) {
        std::string filepath = entry.path().string();
        if(entry.is_directory()) {
            std::string filename = entry.path().filename().string();
            const uint64_t oldSize = this->paths.size();
            lifuren::files::listFiles(this->paths, entry.path().string(), exts);
            const uint64_t newSize = this->paths.size();
            for(uint64_t index = oldSize; index < newSize; ++index) {
                this->labels.push_back(mapping.at(filename));
            }
        } else {
            SPDLOG_DEBUG("忽略无效文件：{}", filepath);
        }
    }
}

torch::optional<size_t> lifuren::datasets::FileDataset::size() const {
    return this->paths.size();
}

torch::data::Example<> lifuren::datasets::FileDataset::get(size_t index) {
    const std::string& path   = this->paths.at(index);
    torch::Tensor data = this->fileTransform(path);
    const int label = this->labels.at(index);
    torch::Tensor target = torch::full({1}, label);
    return { 
        data,
        target
    };
}

lifuren::datasets::TensorDataset::TensorDataset(
    torch::Tensor& features,
    torch::Tensor& labels
) : features(features), labels(labels) {
}

torch::optional<size_t> lifuren::datasets::TensorDataset::size() const {
    return this->features.sizes()[0];
    // return this->labels.sizes[0];
}

torch::data::Example<> lifuren::datasets::TensorDataset::get(size_t index) {
    return {
        this->features[index],
        this->labels[index]
    };
}

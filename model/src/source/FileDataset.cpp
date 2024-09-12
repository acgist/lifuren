#include "lifuren/Datasets.hpp"

#include "ggml.h"

#include "spdlog/spdlog.h"

#include "lifuren/Files.hpp"

static void listFile(
    const std::string& path,
    const std::vector<std::string>& exts,
    std::function<void(const std::string&, std::vector<std::vector<float>>&)> transform,
    std::vector<std::vector<float>>& features
);

static void listMappingFile(
    const std::string& path,
    const std::vector<std::string>& exts,
    const std::map<std::string, float>& mapping,
    std::function<void(const std::string&, std::vector<std::vector<float>>&)> transform,
    std::vector<std::vector<float>>& features,
    std::vector<float>& labels
);

lifuren::datasets::FileDataset::FileDataset(
    const size_t& batchSize,
    const std::string& path,
    const std::vector<std::string>& exts,
    std::function<void(const std::string&, std::vector<std::vector<float>>&)> transform,
    const std::map<std::string, float>& mapping
) : lifuren::datasets::Dataset(batchSize), transform(transform) {
    if(!lifuren::files::exists(path) || !lifuren::files::isDirectory(path)) {
        SPDLOG_DEBUG("目录无效：{}", path);
        return;
    }
    if(mapping.empty()) {
        listFile(path, exts, this->transform, this->features);
    } else {
        listMappingFile(path, exts, mapping, this->transform, this->features, this->labels);
    }
    this->count = this->features.size();
}

lifuren::datasets::FileDataset::~FileDataset() {

}

size_t lifuren::datasets::FileDataset::batchGet(size_t batch, void* features, void* labels) const {
    const size_t begin = batch * this->batchSize;
    const size_t end   = std::min(this->count, begin + this->batchSize);
    if(end <= begin) {
        return 0;
    }
    for(size_t index = begin; index < end; ++index) {
        const auto& feature = this->features[index];
        const size_t featureSize = feature.size() * sizeof(float);
        memcpy(features, feature.data(), featureSize);
        if(!this->labels.empty()) {
            const auto& label = this->labels[index];
            memcpy(labels, &label, sizeof(float));
        }
    }
    return end - begin;
}

static void listFile(
    const std::string& path,
    const std::vector<std::string>& exts,
    std::function<void(const std::string&, std::vector<std::vector<float>>&)> transform,
    std::vector<std::vector<float>>& features
) {
    std::vector<std::string> files;
    lifuren::files::listFiles(files, path, exts);
    for(const auto& file : files) {
        transform(file, features);
    }
}

static void listMappingFile(
    const std::string& path,
    const std::vector<std::string>& exts,
    const std::map<std::string, float>& mapping,
    std::function<void(const std::string&, std::vector<std::vector<float>>&)> transform,
    std::vector<std::vector<float>>& features,
    std::vector<float>& labels
) {
    std::vector<std::string> files;
    const auto iterator = std::filesystem::directory_iterator(std::filesystem::u8path(path));
    for(const auto& entry : iterator) {
        files.clear();
        // TODO: utf8
        const std::string filepath = entry.path().string();
        if(entry.is_directory()) {
            lifuren::files::listFiles(files, filepath, exts);
            const std::string filename = entry.path().filename().string();
            for(const auto& file : files) {
                transform(file, features);
            }
            labels.resize(features.size(), mapping.at(filename));
        } else {
            SPDLOG_DEBUG("忽略无效文件：{}", filepath);
        }
    }
}

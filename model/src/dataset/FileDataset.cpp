#include "lifuren/Dataset.hpp"

#include <random>
#include <algorithm>

#include "spdlog/spdlog.h"

#include "lifuren/File.hpp"

static void listFile(
    const std::string& path,
    const std::vector<std::string>& exts,
    std::function<void(const std::string&, std::vector<std::vector<float>>&)> transform,
    std::vector<std::vector<float>>& features,
    bool shuffle
);

static void listMappingFile(
    const std::string& path,
    const std::vector<std::string>& exts,
    const std::map<std::string, float>& mapping,
    std::function<void(const std::string&, std::vector<std::vector<float>>&)> transform,
    std::vector<std::vector<float>>& features,
    std::vector<float>& labels,
    bool shuffle
);

lifuren::dataset::FileDataset::FileDataset(
    const size_t& batchSize,
    const std::string& path,
    const std::vector<std::string>& exts,
    std::function<void(const std::string&, std::vector<std::vector<float>>&)> transform,
    const std::map<std::string, float>& mapping,
    bool shuffle
) : lifuren::dataset::Dataset(batchSize), transform(transform)
{
    if(!lifuren::file::exists(path) || !lifuren::file::isDirectory(path)) {
        SPDLOG_DEBUG("目录无效：{}", path);
        return;
    }
    if(mapping.empty()) {
        listFile(path, exts, this->transform, this->features, shuffle);
    } else {
        listMappingFile(path, exts, mapping, this->transform, this->features, this->labels, shuffle);
    }
    this->count = this->features.size();
}

lifuren::dataset::FileDataset::~FileDataset() {

}

size_t lifuren::dataset::FileDataset::batchGet(size_t batch, float* features, float* labels) const {
    if(features == nullptr && labels == nullptr) {
        return 0LL;
    }
    const size_t beg = batch * this->batchSize;
    const size_t end = std::min(this->count, beg + this->batchSize);
    if(end <= beg || this->features.empty()) {
        return 0LL;
    }
    size_t pos = 0LL;
    for(size_t index = beg; index < end; ++index) {
        const auto& feature = this->features[index];
        std::copy(feature.begin(), feature.end(), features + pos);
        pos += feature.size();
    }
    if(labels != nullptr && !this->labels.empty()) {
        std::copy(this->labels.begin() + beg, this->labels.begin() + end, labels);
    }
    return end - beg;
}

static void listFile(
    const std::string& path,
    const std::vector<std::string>& exts,
    std::function<void(const std::string&, std::vector<std::vector<float>>&)> transform,
    std::vector<std::vector<float>>& features,
    bool shuffle
) {
    std::vector<std::string> files;
    lifuren::file::listFile(files, path, exts);
    if(shuffle) {
        std::default_random_engine random(std::random_device{}());
        std::shuffle(files.begin(), files.end(), random);
    }
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
    std::vector<float>& labels,
    bool shuffle
) {
    std::vector<std::string> files;
    std::vector<std::pair<std::string, float>> fileLabelMapping;
    const auto iterator = std::filesystem::directory_iterator(std::filesystem::u8path(path));
    for(const auto& entry : iterator) {
        files.clear();
        // TODO: utf8
        const std::string filepath = entry.path().string();
        if(entry.is_directory()) {
            lifuren::file::listFile(files, filepath, exts);
            const std::string filename = entry.path().filename().string();
            for(const auto& file : files) {
                fileLabelMapping.emplace_back(std::make_pair(file, mapping.at(filename)));
            }
        } else {
            SPDLOG_DEBUG("忽略无效文件：{}", filepath);
        }
    }
    if(shuffle) {
        std::default_random_engine random(std::random_device{}());
        std::shuffle(std::begin(fileLabelMapping), std::end(fileLabelMapping), random);
    }
    features.reserve(fileLabelMapping.size());
    labels.reserve(fileLabelMapping.size());
    for(const auto& [file, label] : fileLabelMapping) {
        transform(file, features);
        labels.push_back(label);
    }
}

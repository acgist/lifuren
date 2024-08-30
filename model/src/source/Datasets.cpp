#include "lifuren/Datasets.hpp"

#include "ggml.h"

#include "spdlog/spdlog.h"

#include "lifuren/Files.hpp"

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

ggml_tensor* lifuren::datasets::readImage(const std::string& path, int width, int height, const std::function<void(const cv::Mat&)> imageTransform) {
    cv::Mat image = cv::imread(path);
    cv::resize(image, image, cv::Size(width, height));
    if(imageTransform != nullptr) {
        imageTransform(image);
    }
    image.release();
    return nullptr;
}

lifuren::datasets::Dataset::Dataset(size_t batchSize) : batchSize(batchSize) {
}

lifuren::datasets::Dataset::Dataset(size_t count, size_t batchSize) : count(count), batchSize(batchSize) {
}

lifuren::datasets::Dataset::~Dataset() {
}

size_t lifuren::datasets::Dataset::getCount() const {
    return this->count;
}

size_t lifuren::datasets::Dataset::getBatchSize() const {
    return this->batchSize;
}

size_t lifuren::datasets::Dataset::getBatchCount() const {
    if(this->count % this->batchSize == 0) {
        return this->count / this->batchSize;
    } else {
        return (this->count / this->batchSize) + 1;
    }
}

lifuren::datasets::FileDataset::FileDataset(
    size_t batchSize,
    const std::string& path,
    const std::vector<std::string>& exts,
    const std::map<std::string, int>& mapping,
    const std::function<ggml_tensor*(const std::string&)> fileTransform
) : lifuren::datasets::Dataset(batchSize), fileTransform(fileTransform) {
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

size_t lifuren::datasets::FileDataset::batchGet(size_t index, void* datas, size_t maxDataSize, void* labels, size_t maxLabelSize) const {
    // memcpy(model.images->data, images + iex0*MNIST_NINPUT,   ggml_nbytes(model.images));
    // memcpy(model.labels->data, labels + iex0*MNIST_NCLASSES, ggml_nbytes(model.labels));
    return 0;
}

lifuren::datasets::TensorDataset::TensorDataset(
    size_t count,
    size_t batchSize,
    float* features,
    size_t feature_size,
    float* labels,
    size_t label_size
) : lifuren::datasets::Dataset(count, batchSize), features(features), feature_size(feature_size), labels(labels), label_size(label_size) {
}

size_t lifuren::datasets::TensorDataset::batchGet(size_t index, void* datas, size_t maxDataSize, void* labels, size_t maxLabelSize) const {
    const size_t remaining = this->count - this->batchSize * index;
    if(remaining >= this->batchSize) {
        memcpy(datas,  this->features + index * this->batchSize * feature_size, maxDataSize);
        memcpy(labels, this->labels   + index * this->batchSize * label_size,   maxLabelSize);
        return this->batchSize;
    } else {
        memcpy(datas,  this->features + index * this->batchSize * feature_size, sizeof(float) * remaining * feature_size);
        memcpy(labels, this->labels   + index * this->batchSize * label_size,   sizeof(float) * remaining * label_size);
        return remaining;
    }
}

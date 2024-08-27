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

lifuren::datasets::FileDataset::FileDataset(
    const std::string& path,
    const std::vector<std::string>& exts,
    const std::map<std::string, int>& mapping,
    const std::function<ggml_tensor*(const std::string&)> fileTransform
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

size_t lifuren::datasets::FileDataset::size() const {
    return this->paths.size();
}

ggml_tensor* lifuren::datasets::FileDataset::get(size_t index) {
    const std::string& path = this->paths.at(index);
    // TODO
    return nullptr;
}

lifuren::datasets::TensorDataset::TensorDataset(
    ggml_tensor* features,
    ggml_tensor* labels
) : features(features), labels(labels) {
}

size_t lifuren::datasets::TensorDataset::size() const {
    // TODO
    return 0;
}

ggml_tensor* lifuren::datasets::TensorDataset::get(size_t index) {
    return nullptr;
}

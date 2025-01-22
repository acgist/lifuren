#include "lifuren/image/ImageDataset.hpp"

#include "lifuren/image/Image.hpp"

torch::Tensor lifuren::dataset::image::feature(const int& width, const int& height, const std::string& file, const torch::DeviceType& type) {
    size_t length{ 0 };
    std::vector<char> feature;
    feature.resize(width * height * 3);
    lifuren::image::read(file, feature.data(), width, height);
    return std::move(lifuren::dataset::image::feature(feature.data(), width, height, type));
}

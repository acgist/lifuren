#include "lifuren/image/ImageDataset.hpp"

torch::Tensor lifuren::dataset::image::feature(const int& width, const int& height, const std::string& file, const torch::DeviceType& type) {
    size_t length{ 0 };
    std::vector<float> feature;
    feature.resize(width * height * 3);
    lifuren::image::load(file, feature.data(), length, width, height, nullptr);
    return torch::from_blob(feature.data(), { height, width, 3 }, torch::kByte).permute({2, 0, 1}).to(torch::kF32).div(255.0).to(type);
}

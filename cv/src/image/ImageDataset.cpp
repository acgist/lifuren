#include "lifuren/image/ImageDataset.hpp"

#include "spdlog/spdlog.h"

#include "lifuren/File.hpp"

#include "opencv2/opencv.hpp"

bool lifuren::image::read(const std::string& path, char* data, const size_t& width, const size_t& height) {
    auto image{ cv::imread(path) };
    return lifuren::image::read(image, data, width, height);
}

bool lifuren::image::read(cv::Mat& image, char* data, const size_t& width, const size_t& height) {
    if(image.total() <= 0LL) {
        return false;
    }
    cv::resize(image, image, cv::Size(width, height));
    std::memcpy(data, image.data, image.total() * image.elemSize());
    return true;
}

bool lifuren::image::write(const std::string& path, const char* data, const size_t& width, const size_t& height) {
    if(data == nullptr) {
        return false;
    }
    if(width == 0LL || height == 0LL) {
        return false;
    }
    lifuren::file::createParent(path);
    cv::Mat image(height, width, CV_8UC3);
    std::memcpy(image.data, data, width * height * 3);
    return cv::imwrite(path, image);
}

torch::Tensor lifuren::image::feature(const int& width, const int& height, const std::string& file, const torch::DeviceType& type) {
    size_t length{ 0 };
    std::vector<char> feature;
    feature.resize(width * height * 3);
    lifuren::image::read(file, feature.data(), width, height);
    return std::move(lifuren::image::feature(feature.data(), width, height, type));
}

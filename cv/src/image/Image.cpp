#include "lifuren/image/Image.hpp"

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

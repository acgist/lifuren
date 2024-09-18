#include "lifuren/Images.hpp"

#include "spdlog/spdlog.h"

#include "lifuren/Files.hpp"

#include "opencv2/opencv.hpp"

bool lifuren::images::read(const std::string& path, uint8_t** data, size_t& width, size_t& height, size_t& length) {
    cv::Mat image{ cv::imread(path) };
    if(image.total() <= 0) {
        return false;
    }
    width  = image.cols;
    height = image.rows;
    length = image.total() * image.elemSize();
    *data  = new uint8_t[length];
    std::memcpy(*data, image.data, length);
    return true;
}

bool lifuren::images::write(const std::string& path, uint8_t* data, size_t width, size_t height, size_t length, size_t channel) {
    if(data == nullptr) {
        return false;
    }
    if(length <= 0) {
        length = width * height * channel;
    }
    if(width <= 0 || height <= 0 || length <= 0) {
        return false;
    }
    lifuren::files::createParent(path);
    cv::Mat image(height, width, channel == 1LL ? CV_8UC1 :
                                 channel == 2LL ? CV_8UC2 :
                                 channel == 3LL ? CV_8UC3 : CV_8UC4);
    memcpy(image.data, data, length);
    bool success = cv::imwrite(path, image);
    return success;
}

void lifuren::images::show(uint8_t* data, size_t width, size_t height, size_t length) {
    ::cv::Mat image(static_cast<int>(height), static_cast<int>(width), CV_8UC3);
    memcpy(image.data, data, length);
    ::cv::imshow("lifuren_show", image);
    ::cv::waitKey();
}

void lifuren::images::load(
    const std::string& path,
    float * data,
    size_t& length,
    const int& width,
    const int& height,
    const std::function<void(const cv::Mat&)> transform
) {
    const cv::Mat image = cv::imread(path);
    if(image.channels() <= 0) {
        SPDLOG_WARN("图片读取失败：{}", path);
        return;
    }
    const cv::Mat target(height, width, CV_8UC3);
    if(width > 0 && height > 0) {
        cv::resize(image, target, cv::Size(width, height));
    }
    if(transform != nullptr) {
        transform(target);
    }
    length = target.total() * target.elemSize();
    std::copy(target.data, target.data + length, data);
    // std::transform(target.data, target.data + length, data, [](const auto& v) { return static_cast<float>(v); });
}

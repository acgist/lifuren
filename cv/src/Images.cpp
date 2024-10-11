#include "lifuren/Images.hpp"

#include "spdlog/spdlog.h"

#include "lifuren/Files.hpp"

#include "opencv2/opencv.hpp"

bool lifuren::images::read(const std::string& path, uint8_t** data, size_t& width, size_t& height, size_t& length) {
    cv::Mat image{ cv::imread(path) };
    if(image.total() <= 0LL) {
        return false;
    }
    width  = image.cols;
    height = image.rows;
    length = image.total() * image.elemSize();
    *data  = new uint8_t[length];
    std::memcpy(*data, image.data, length);
    return true;
}

bool lifuren::images::write(const std::string& path, const uint8_t* data, const size_t& width, const size_t& height, size_t length, const size_t& channel) {
    if(data == nullptr) {
        return false;
    }
    if(width <= 0LL || height <= 0LL || channel <= 0LL) {
        return false;
    }
    if(length <= 0LL) {
        length = width * height * channel;
    }
    lifuren::files::createParent(path);
    cv::Mat image(height, width, channel == 1LL ? CV_8UC1 :
                                 channel == 2LL ? CV_8UC2 :
                                 channel == 3LL ? CV_8UC3 : CV_8UC4);
    std::memcpy(image.data, data, length);
    return cv::imwrite(path, image);
}

bool lifuren::images::resize(uint8_t* input, size_t width, size_t height, uint8_t* output, size_t output_width, size_t output_height) {
    const size_t input_length = width * height * 3;
    if(width == output_width && height == output_height) {
        std::copy(input, input + input_length, output);
        return true;
    }
    cv::Mat source(static_cast<int>(height),        static_cast<int>(width),        CV_8UC3, input);
    cv::Mat target(static_cast<int>(output_height), static_cast<int>(output_width), CV_8UC3, output);
    cv::resize(source, target, cv::Size(output_width, output_height));
    return true;
}

void lifuren::images::show(const uint8_t* data, const size_t& width, const size_t& height, const size_t& length) {
    cv::Mat image(static_cast<int>(height), static_cast<int>(width), CV_8UC3);
    std::memcpy(image.data, data, length == 0LL ? width * height * 3 : length);
    cv::imshow("lifuren_show", image);
    cv::waitKey();
}

void lifuren::images::load(
    const std::string& path,
    float * data,
    size_t& length,
    const size_t& width,
    const size_t& height,
    const std::function<void(const cv::Mat&)> transform
) {
    const cv::Mat image = cv::imread(path);
    if(image.total() <= 0LL) {
        SPDLOG_WARN("图片读取失败：{}", path);
        return;
    }
    if(width > 0LL && height > 0LL) {
        cv::resize(image, image, cv::Size(width, height));
    }
    if(transform != nullptr) {
        transform(image);
    }
    length = image.total() * image.elemSize();
    std::copy(image.data, image.data + length, data);
    // std::transform(image.data, image.data + length, data, [](const auto& v) { return static_cast<float>(v); });
}

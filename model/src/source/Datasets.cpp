#include "lifuren/Datasets.hpp"

#include "opencv2/opencv.hpp"

void lifuren::datasets::readImage(
    const std::string& path,
    float * data,
    size_t& length,
    const int& width,
    const int& height,
    const std::function<void(const cv::Mat&)> transform
) {
    const cv::Mat image = cv::imread(path);
    if(width > 0 && height > 0) {
        cv::resize(image, image, cv::Size(width, height));
    }
    if(transform != nullptr) {
        transform(image);
    }
    length = image.total() * image.elemSize();
    std::copy(image.data, image.data + length, data);
    // 析构调用可以忽略
    // image.release();
}

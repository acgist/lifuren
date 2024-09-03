#include "lifuren/Images.hpp"

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
    image.release();
    return true;
}

bool lifuren::images::write(const std::string& path, uint8_t* data, size_t width, size_t height, size_t length) {
    if(data == nullptr) {
        return false;
    }
    if(length <= 0) {
        length = width * height * 3;
    }
    if(width <= 0 || height <= 0 || length <= 0) {
        return false;
    }
    cv::Mat image(height, width, CV_8UC3);
    memcpy(image.data, data, length);
    bool success = cv::imwrite(path, image);
    image.release();
    return success;
}

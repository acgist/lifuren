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
    cv::Mat image(height, width, channel == 1LL ? CV_8UC1 :
                                 channel == 2LL ? CV_8UC2 :
                                 channel == 3LL ? CV_8UC3 : CV_8UC4);
    memcpy(image.data, data, length);
    bool success = cv::imwrite(path, image);
    image.release();
    return success;
}

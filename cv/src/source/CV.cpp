#include "lifuren/CV.hpp"

#include "opencv2/opencv.hpp"
#include "opencv2/core/utils/logger.hpp"

void lifuren::cv::logger() {
    #if defined(_DEBUG) || !defined(NDEBUG)
    ::cv::utils::logging::setLogLevel(::cv::utils::logging::LOG_LEVEL_DEBUG);
    #else
    ::cv::utils::logging::setLogLevel(::cv::utils::logging::LOG_LEVEL_INFO);
    #endif
}

void lifuren::cv::show(uint8_t* data, size_t width, size_t height, size_t length) {
    ::cv::Mat image(static_cast<int>(height), static_cast<int>(width), CV_8UC3);
    memcpy(image.data, data, length);
    ::cv::imshow("cv_show", image);
    ::cv::waitKey();
    image.release();
}

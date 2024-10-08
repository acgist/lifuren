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

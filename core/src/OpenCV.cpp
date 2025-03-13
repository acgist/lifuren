#include "lifuren/Logger.hpp"

#include "opencv2/core/utils/logger.hpp"

void lifuren::logger::opencv::init() {
    #if defined(_DEBUG) || !defined(NDEBUG)
    cv::utils::logging::setLogLevel(::cv::utils::logging::LOG_LEVEL_DEBUG);
    #else
    cv::utils::logging::setLogLevel(::cv::utils::logging::LOG_LEVEL_INFO);
    #endif
}

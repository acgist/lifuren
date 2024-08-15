#include "lifuren/CV.hpp"

#include "opencv2/core/utils/logger.hpp"

void lifuren::cv::logger() {
    // 关闭OpenCV日志
    ::cv::utils::logging::setLogLevel(::cv::utils::logging::LOG_LEVEL_WARNING);
    // ::cv::utils::logging::setLogLevel(::cv::utils::logging::LOG_LEVEL_ERROR);
    // ::cv::utils::logging::setLogLevel(::cv::utils::logging::LOG_LEVEL_FATAL);
    // ::cv::utils::logging::setLogLevel(::cv::utils::logging::LOG_LEVEL_SILENT);
}

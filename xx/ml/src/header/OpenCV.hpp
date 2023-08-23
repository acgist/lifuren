#pragma once

#include "opencv2/core/utils/logger.hpp"

namespace lifuren {

/**
 * 关闭日志
 */
void offOpenCVLogin() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
}
    
}
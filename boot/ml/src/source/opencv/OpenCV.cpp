#include "../../header/OpenCV.hpp"

void lifuren::shutdownOpenCVLogger() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
}
#include "header/DLibAll.hpp"
#include "header/LifurenGG.hpp"
#include "opencv2/core/utils/logger.hpp"

int main(int argc, char const* argv[]) {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    lifuren::gg::init(argc, argv);
    // testGLog(argc, argv);
    // testJson();
    // testString();
    // lifuren::ml::test3DPointCloud();
    // lifuren::ml::testBase64Encoder();
    lifuren::ml::testLinearRegression();
    return 0;
}

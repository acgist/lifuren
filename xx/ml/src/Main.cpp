#include "header/DLibAll.hpp"
#include "header/LifurenGG.hpp"
#include "header/OpenCV.hpp"

int main(int argc, char const* argv[]) {
    lifuren::ml::offOpenCVLogin();
    lifuren::gg::init(argc, argv);
    // lifuren::ml::testGLog(argc, argv);
    // lifuren::ml::testJson();
    // lifuren::ml::testString();
    // lifuren::ml::test3DPointCloud();
    // lifuren::ml::testBase64Encoder();
    lifuren::ml::testLinearRegression();
    return 0;
}

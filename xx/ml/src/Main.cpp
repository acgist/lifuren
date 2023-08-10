#include "header/DLibAll.hpp"
#include "header/LifurenGG.hpp"

int main(int argc, char const* argv[]) {
    lifuren::gg::init(argc, argv);
    // testGLog(argc, argv);
    // testJson();
    // testString();
    // lifuren::ml::test3DPointCloud();
    // lifuren::ml::testBase64Encoder();
    lifuren::ml::testLinearRegression();
    return 0;
}

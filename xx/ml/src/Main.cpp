#include "header/DLibAll.hpp"
#include "header/LifurenGG.hpp"
#include "header/MathGL.hpp"

int main(int argc, char const* argv[]) {
    lifuren::gg::init(argc, argv);
    // testGLog(argc, argv);
    // testJson();
    // testString();
    // lifuren::ml::test3DPointCloud();
    // lifuren::ml::testBase64Encoder();
    // lifuren::ml::testLinearRegression();
    lifuren::ml::draw();
    return 0;
}

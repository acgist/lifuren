#include "header/Boot.hpp"
#include "header/Fltk.hpp"
#include "header/MLPack.hpp"
#include "header/OpenCV.hpp"
#include "header/LibTorch.hpp"

int main(int argc, char const* argv[]) {
    lifuren::init(argc, argv);
    lifuren::shutdownOpenCVLogger();
    LOG(INFO) << "测试";
    // lifuren::testJson();
    // lifuren::testMark();
    // lifuren::testLabel();
    // lifuren::matrix();
    lifuren::linearRegression();
    // lifuren::testPlus();
    // lifuren::testLinear();
    // lifuren::testReLU();
    // lifuren::testTanh();
    LOG(INFO) << "完成";
    lifuren::shutdown();
    return 0;
}

#include "header/Boot.hpp"
#include "header/Fltk.hpp"
#include "header/MLPack.hpp"
#include "header/OpenCV.hpp"
#include "header/LibTorch.hpp"

int main(int argc, char const* argv[]) {
    lifuren::init(argc, argv);
    LOG(INFO) << "测试";
    // lifuren
    // lifuren::testJson();
    // lifuren::testMark();
    // lifuren::testLabel();
    // opencv
    // lifuren::shutdownOpenCVLogger();
    // mlpack
    // lifuren::testLoadFile();
    // lifuren::testMLPackMatrix();
    // lifuren::testMLPackLinearRegression();
    lifuren::testMLPackLogisticRegression();
    // libtorch
    // lifuren::testMatrix();
    // lifuren::testReLU();
    // lifuren::testTanh();
    // lifuren::testLinearRegression();
    LOG(INFO) << "完成";
    // lifuren::LifurenWindow* windowPtr = new lifuren::LifurenWindow(200, 100, "李夫人");
    // windowPtr->init();
    // windowPtr->show();
    // const int ret = Fl::run();
    lifuren::shutdown();
    // return ret;
    return 0;
}

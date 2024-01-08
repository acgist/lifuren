#include "../src/header/OpenCV.hpp"

int main(const int argc, const char * const argv[]) {
    lifuren::init(argc, argv);
    LOG(INFO) << "测试";
    lifuren::shutdownOpenCVLogger();
    lifuren::color("D:/tmp/logo.png");
    // lifuren::color("D:/tmp/Dota2.jpg");
    LOG(INFO) << "完成";
    lifuren::shutdown();
    return 0;
}

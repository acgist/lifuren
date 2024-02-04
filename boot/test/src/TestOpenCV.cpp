#include "../src/header/OpenCV.hpp"

int main(const int argc, const char * const argv[]) {
    lifuren::init(argc, argv);
    LOG(INFO) << "测试";
    lifuren::shutdownOpenCVLogger();
    lifuren::readAndShow("D:/tmp/Dota2.jpg");
    LOG(INFO) << "完成";
    lifuren::shutdown();
    return 0;
}

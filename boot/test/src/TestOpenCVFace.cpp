#include "../src/header/OpenCV.hpp"

int main(const int argc, const char * const argv[]) {
    lifuren::init(argc, argv);
    LOG(INFO) << "测试";
    lifuren::shutdownOpenCVLogger();
    lifuren::face("D:/gitee/lifuren/deps/opencv/etc/haarcascades/haarcascade_frontalface_default.xml", "D:/tmp/F4.jpg");
    // lifuren::face("D:/gitee/lifuren/deps/opencv/etc/haarcascades/haarcascade_frontalface_default.xml", "D:/tmp/head.jpg");
    LOG(INFO) << "完成";
    lifuren::shutdown();
    return 0;
}

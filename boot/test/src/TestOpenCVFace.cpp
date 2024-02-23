#include "../src/header/OpenCV.hpp"

int main(const int argc, const char * const argv[]) {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    lifuren::shutdownOpenCVLogger();
    lifuren::face("D:/gitee/lifuren/deps/opencv/etc/haarcascades/haarcascade_frontalface_default.xml", "D:/tmp/F4.jpg");
    // lifuren::face("D:/gitee/lifuren/deps/opencv/etc/haarcascades/haarcascade_frontalface_default.xml", "D:/tmp/head.jpg");
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}

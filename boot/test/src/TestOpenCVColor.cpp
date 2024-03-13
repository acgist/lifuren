#include "./header/OpenCV.hpp"

int main(const int argc, const char * const argv[]) {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    lifuren::shutdownOpenCVLogger();
    lifuren::color("D:/tmp/logo.png");
    // lifuren::color("D:/tmp/Dota2.jpg");
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}

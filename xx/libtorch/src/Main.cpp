#include "header/LibTorch.hpp"

int main(int argc, char const *argv[]) {
    lifuren::initGlog(argc, argv);
    LOG(INFO) << "测试";
    // lifuren::testPlus();
    lifuren::testLinear();
    LOG(INFO) << "完成";
    lifuren::shutdownGlog();
    return 0;
}

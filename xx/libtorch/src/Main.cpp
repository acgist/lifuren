#include "header/LibTorch.hpp"

int main(int argc, char const *argv[]) {
    lifuren::glog::init(argc, argv);
    LOG(INFO) << "测试";
    // lifuren::testPlus();
    lifuren::testLinear();
    LOG(INFO) << "完成";
    lifuren::glog::shutdown();
    return 0;
}

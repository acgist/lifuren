#include "header/CUDA.hpp"

int main(const int argc, const char * const argv[]) {
    lifuren::init(argc, argv);
    LOG(INFO) << "测试";
    lifuren::testCUDA();
    LOG(INFO) << "完成";
    lifuren::shutdown();
    return 0;
}
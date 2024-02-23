#include "../src/header/CNN.hpp"

int main(const int argc, const char * const argv[]) {
    lifuren::init(argc, argv);
    LOG(INFO) << "测试";
    lifuren::testMNIST();
    LOG(INFO) << "完成";
    lifuren::shutdown();
    return 0;
}

#include "../src/header/Core.hpp"

int main(const int argc, const char * const argv[]) {
    lifuren::init(argc, argv);
    LOG(INFO) << "测试";
    lifuren::testJson();
    lifuren::testMark();
    lifuren::testLabel();
    lifuren::testSetting();
    LOG(INFO) << "完成";
    lifuren::shutdown();
    return 0;
}

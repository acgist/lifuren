#include "./header/Core.hpp"

int main(const int argc, const char * const argv[]) {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    lifuren::testJson();
    lifuren::testMark();
    lifuren::testLabel();
    lifuren::testSetting();
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}

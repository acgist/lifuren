#include "../src/header/CNN.hpp"

int main(const int argc, const char * const argv[]) {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    lifuren::testVGG();
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}

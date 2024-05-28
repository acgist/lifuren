
#include "../../header/LibTorch.hpp"

void lifuren::testYOLO() {
}

int main(const int argc, const char * const argv[]) {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    lifuren::testYOLO();
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}

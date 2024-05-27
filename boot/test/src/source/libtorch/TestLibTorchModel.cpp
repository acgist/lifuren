
#include "../../header/LibTorch.hpp"

int main(const int argc, const char * const argv[]) {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    try {
        lifuren::testModel();
    } catch(const std::exception& e) {
        SPDLOG_ERROR("加载模型异常：{}", e.what());
    }
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}

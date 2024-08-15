#include "lifuren/Exception.hpp"

#include "lifuren/Logger.hpp"

#include "spdlog/spdlog.h"

static void testException() {
    try {
        int* ptr{ nullptr };
        // lifuren::Exception::trueThrow(ptr);
        lifuren::Exception::falseThrow(ptr, lifuren::CODE_1000, "兄弟报错了🤡");
        // lifuren::Exception::throwException();
    } catch(const std::exception& e) {
        SPDLOG_DEBUG("异常：{}", e.what());
    }
}

int main() {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    testException();
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}
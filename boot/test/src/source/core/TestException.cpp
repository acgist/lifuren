#include "Test.hpp"

#include "lifuren/Exception.hpp"

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

LFR_TEST(
    testException();
);

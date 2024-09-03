#include "lifuren/Images.hpp"

#include "spdlog/spdlog.h"

#include "lifuren/CV.hpp"
#include "lifuren/Logger.hpp"

static void testRead() {
    uint8_t* data{nullptr};
    size_t width {0};
    size_t height{0};
    size_t length{0};
    // lifuren::images::read("D:/tmp/fail.jpg", &data, width, height, length);
    lifuren::images::read("D:/tmp/girl.png", &data, width, height, length);
    lifuren::cv::show(data, width, height, length);
    delete data;
    data = nullptr;
}

static void testWrite() {
    uint8_t* data{nullptr};
    size_t width {0};
    size_t height{0};
    size_t length{0};
    bool success = lifuren::images::read("D:/tmp/girl.png", &data, width, height, length);
    if(success) {
        lifuren::images::write("D:/tmp/girl_copy.png", data, width, height, length);
    }
    delete data;
    data = nullptr;
}

int main(const int argc, const char * const argv[]) {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    testRead();
    // testWrite();
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}

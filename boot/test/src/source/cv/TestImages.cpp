#include "Test.hpp"
#include "lifuren/Images.hpp"

#include "lifuren/CV.hpp"

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

LFR_TEST(
    testRead();
    // testWrite();
);

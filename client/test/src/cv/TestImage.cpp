#include "lifuren/Test.hpp"

#include "opencv2/opencv.hpp"

#include "lifuren/File.hpp"
#include "lifuren/image/Image.hpp"

[[maybe_unused]] static void testRead() {
    size_t width { 128 };
    size_t height{ 128 };
    cv::Mat image(width, height, CV_8UC3);
    const bool success = lifuren::image::read(lifuren::file::join({lifuren::config::CONFIG.tmp, "girl.png"}).string(), reinterpret_cast<char*>(image.data), width, height);
    if(success) {
        cv::imshow("image", image);
        cv::waitKey(0);
    }
}

[[maybe_unused]] static void testWrite() {
    size_t width { 128 };
    size_t height{ 128 };
    cv::Mat image(width, height, CV_8UC3);
    const bool success = lifuren::image::read(lifuren::file::join({lifuren::config::CONFIG.tmp, "girl.png"}).string(), reinterpret_cast<char*>(image.data), width, height);
    if(success) {
        lifuren::image::write(lifuren::file::join({lifuren::config::CONFIG.tmp, "girl_copy.png"}).string(), reinterpret_cast<char*>(image.data), width, height);
    }
}

LFR_TEST(
    testRead();
    // testWrite();
);

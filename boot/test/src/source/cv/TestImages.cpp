#include "Test.hpp"

#include "lifuren/CV.hpp"
#include "lifuren/Images.hpp"

#include "opencv2/opencv.hpp"

[[maybe_unused]] static void testRead() {
    uint8_t* data{nullptr};
    size_t width {0};
    size_t height{0};
    size_t length{0};
    // lifuren::images::read("D:/tmp/fail.jpg", &data, width, height, length);
    lifuren::images::read("D:/tmp/girl.png", &data, width, height, length);
    lifuren::images::show(data, width, height, length);
    delete data;
    data = nullptr;
}

[[maybe_unused]] static void testWrite() {
    uint8_t* data{nullptr};
    size_t width {0};
    size_t height{0};
    size_t length{0};
    bool success = lifuren::images::read("D:/tmp/girl.jpg", &data, width, height, length);
    if(success) {
        lifuren::images::write("D:/tmp/girl_copy.png", data, width, height, length);
        uint8_t* x = new uint8_t[width * height];
        for(size_t i = 0; i < width * height; ++i) {
            x[i] = data[3 * i];
        }
        lifuren::images::write("D:/tmp/girl_1.png", x, width, height, 0, 1);
        for(size_t i = 0; i < width * height; ++i) {
            x[i] = data[3 * i + 1];
        }
        lifuren::images::write("D:/tmp/girl_2.png", x, width, height, 0, 1);
        for(size_t i = 0; i < width * height; ++i) {
            x[i] = data[3 * i + 2];
        }
        lifuren::images::write("D:/tmp/girl_3.png", x, width, height, 0, 1);
        delete x;
        x = nullptr;
    }
    delete data;
    data = nullptr;
}

[[maybe_unused]] static void testReadTransform() {
    float* data = new float[256 * 256 * 3];
    size_t length{ 0 };
    lifuren::images::readTransform("D:/tmp/logo.png", data, length);
    cv::Mat image(256, 256, CV_8UC3);
    std::copy(data, data + length, image.data);
    // std::transform(data, data + length, image.data, [](const auto& v) { return static_cast<uchar>(v); });
    cv::imwrite("D:/tmp/logo_copy.png", image);
    delete data;
    data = nullptr;
}

[[maybe_unused]] static void testSplit() {
    auto image = cv::imread("D:/tmp/girl.png");
    cv::Mat rgb[3];
    cv::split(image, rgb);
    cv::imshow("rgb-0", rgb[0]);
    cv::waitKey(0);
    cv::imshow("rgb-1", rgb[1]);
    cv::waitKey(0);
    cv::imshow("rgb-2", rgb[2]);
    cv::waitKey(0);
    cv::Mat output;
    cv::merge(rgb, 3, output);
    cv::imshow("rgb", output);
    cv::waitKey(0);
    rgb[0] = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
    cv::merge(rgb, 3, output);
    cv::imshow("rg", output);
    cv::waitKey(0);
}

[[maybe_unused]] static void testMerge() {
    auto a = cv::imread("D:/tmp/girl_1.png");
    auto b = cv::imread("D:/tmp/girl_2.png");
    auto c = cv::imread("D:/tmp/girl_3.png");
    // auto a = cv::imread("D:/tmp/girl_1.png", cv::ImreadModes::IMREAD_GRAYSCALE);
    // auto b = cv::imread("D:/tmp/girl_2.png", cv::ImreadModes::IMREAD_GRAYSCALE);
    // auto c = cv::imread("D:/tmp/girl_3.png", cv::ImreadModes::IMREAD_GRAYSCALE);
    std::vector<cv::Mat> aa(3);
    std::vector<cv::Mat> bb(3);
    std::vector<cv::Mat> cc(3);
    cv::split(a, aa);
    cv::split(b, bb);
    cv::split(c, cc);
    cv::Mat image;
    // cv::Mat image(a.rows, a.cols, CV_8UC3);
    // std::vector<cv::Mat> rgb{ a, b, c };
    std::vector<cv::Mat> rgb{ aa.at(0), bb.at(1), cc.at(2) };
    cv::merge(rgb, image);
    cv::imshow("rgb", image);
    cv::waitKey(0);
}

LFR_TEST(
    // testRead();
    // testWrite();
    // testSplit();
    testMerge();
    // testReadTransform();
);

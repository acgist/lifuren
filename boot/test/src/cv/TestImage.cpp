#include "lifuren/Test.hpp"

#include "lifuren/File.hpp"
#include "lifuren/Image.hpp"

#include "opencv2/opencv.hpp"

[[maybe_unused]] static void testRead() {
    uint8_t* data{nullptr};
    size_t width {0};
    size_t height{0};
    size_t length{0};
    // const bool success = lifuren::image::read(lifuren::file::join({lifuren::config::CONFIG.tmp, "fail.jpg"}).string(), &data, width, height, length);
    const bool success = lifuren::image::read(lifuren::file::join({lifuren::config::CONFIG.tmp, "girl.png"}).string(), &data, width, height, length);
    if(success) {
        lifuren::image::show(data, width, height, length);
    }
    delete[] data;
    data = nullptr;
}

[[maybe_unused]] static void testWrite() {
    uint8_t* data{nullptr};
    size_t width {0};
    size_t height{0};
    size_t length{0};
    const bool success = lifuren::image::read(lifuren::file::join({lifuren::config::CONFIG.tmp, "girl.png"}).string(), &data, width, height, length);
    if(success) {
        lifuren::image::write(lifuren::file::join({lifuren::config::CONFIG.tmp, "girl_copy.png"}).string(), data, width, height, length);
        uint8_t* x = new uint8_t[width * height];
        for(size_t i = 0; i < width * height; ++i) {
            x[i] = data[3 * i];
        }
        lifuren::image::write(lifuren::file::join({lifuren::config::CONFIG.tmp, "girl_1.png"}).string(), x, width, height, 0, 1);
        for(size_t i = 0; i < width * height; ++i) {
            x[i] = data[3 * i + 1];
        }
        lifuren::image::write(lifuren::file::join({lifuren::config::CONFIG.tmp, "girl_2.png"}).string(), x, width, height, 0, 1);
        for(size_t i = 0; i < width * height; ++i) {
            x[i] = data[3 * i + 2];
        }
        lifuren::image::write(lifuren::file::join({lifuren::config::CONFIG.tmp, "girl_3.png"}).string(), x, width, height, 0, 1);
        delete[] x;
        x = nullptr;
    }
    delete[] data;
    data = nullptr;
}

[[maybe_unused]] static void testResize() {
    const size_t data_length = 256 * 256 * 3;
    float* data = new float[data_length];
    size_t length{ 0 };
    lifuren::image::load(lifuren::file::join({lifuren::config::CONFIG.tmp, "logo.png"}).string(), data, length);
    uint8_t* input = new uint8_t[data_length];
    std::copy(data, data + data_length, input);
    size_t width = 200;
    size_t height = 220;
    uint8_t* output = new uint8_t[width * height * 3];
    lifuren::image::resize(input, 256, 256, output, width, height);
    lifuren::image::show(output, width, height);
    lifuren::image::write(lifuren::file::join({lifuren::config::CONFIG.tmp, "logo_resize.png"}).string(), output, width, height);
    delete[] data;
    data = nullptr;
    delete[] input;
    input = nullptr;
    delete[] output;
    output = nullptr;
}

[[maybe_unused]] static void testLoad() {
    float* data = new float[512 * 512 * 3];
    size_t length{ 0 };
    lifuren::image::load(lifuren::file::join({lifuren::config::CONFIG.tmp, "logo.png"}).string(), data, length, 512, 512);
    cv::Mat image(512, 512, CV_8UC3);
    // std::copy(data, data + length, image.data);
    std::transform(data, data + length, image.data, [](const auto& v) { return static_cast<uchar>(v); });
    lifuren::image::show(image.data, image.cols, image.rows, image.total() * image.elemSize());
    cv::imwrite(lifuren::file::join({lifuren::config::CONFIG.tmp, "logo_copy.png"}).string(), image);
    delete[] data;
    data = nullptr;
}

[[maybe_unused]] static void testSplit() {
    auto image = cv::imread(lifuren::file::join({lifuren::config::CONFIG.tmp, "girl.png"}).string());
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
    auto a = cv::imread(lifuren::file::join({lifuren::config::CONFIG.tmp, "girl_1.png"}).string());
    auto b = cv::imread(lifuren::file::join({lifuren::config::CONFIG.tmp, "girl_2.png"}).string());
    auto c = cv::imread(lifuren::file::join({lifuren::config::CONFIG.tmp, "girl_3.png"}).string());
    // auto a = cv::imread(lifuren::file::join({lifuren::config::CONFIG.tmp, "girl_1.png"}).string(), cv::ImreadModes::IMREAD_GRAYSCALE);
    // auto b = cv::imread(lifuren::file::join({lifuren::config::CONFIG.tmp, "girl_2.png"}).string(), cv::ImreadModes::IMREAD_GRAYSCALE);
    // auto c = cv::imread(lifuren::file::join({lifuren::config::CONFIG.tmp, "girl_3.png"}).string(), cv::ImreadModes::IMREAD_GRAYSCALE);
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
    // testResize();
    testLoad();
    // testSplit();
    // testMerge();
);

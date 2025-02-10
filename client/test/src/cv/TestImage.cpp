#include "lifuren/Test.hpp"

#include "opencv2/opencv.hpp"

#include "lifuren/File.hpp"
#include "lifuren/image/Image.hpp"

[[maybe_unused]] static void testFeature() {
    auto image { cv::imread(lifuren::file::join({ lifuren::config::CONFIG.tmp, "xxc.png" }).string()) };
    cv::imshow("image", image);
    cv::waitKey();
    auto tensor = lifuren::image::feature(image, 640, 480, torch::DeviceType::CPU);
    cv::Mat target(480, 640, CV_8UC3);
    lifuren::image::tensor_to_mat(target, tensor);
    cv::imshow("target", target);
    cv::waitKey();
}

LFR_TEST(
    testFeature();
);

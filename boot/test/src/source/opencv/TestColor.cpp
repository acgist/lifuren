#include "../../header/OpenCV.hpp"

#include "Logger.hpp"

#include "spdlog/spdlog.h"

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

void lifuren::color(const std::string& path) {
    cv::Mat image = cv::imread(path);
    cv::imshow("Image", image);
    cv::waitKey(0);
    cv::destroyWindow("Image");
    // 灰度
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_RGB2GRAY);
    cv::imshow("GrayImage", grayImage);
    cv::waitKey(0);
    cv::destroyWindow("GrayImage");
    grayImage.release();
    // 高斯模糊：size必须奇数
    cv::Mat blurImage;
    cv::GaussianBlur(image, blurImage, cv::Size(3, 3), 3, 0);
    cv::imshow("BlurImage", blurImage);
    cv::waitKey(0);
    cv::destroyWindow("BlurImage");
    blurImage.release();
    // 边缘检测
    cv::Mat cannyImage;
    cv::Canny(image, cannyImage, 16, 32);
    cv::imshow("CannyImage", cannyImage);
    cv::waitKey(0);
    cv::destroyWindow("CannyImage");
    cannyImage.release();
    // 图片膨胀
    cv::Mat dilateImage;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(image, dilateImage, kernel);
    cv::imshow("DilateImage", dilateImage);
    cv::waitKey(0);
    cv::destroyWindow("DilateImage");
    dilateImage.release();
    // 图片侵蚀
    cv::Mat erodeImage;
    cv::erode(image, erodeImage, kernel);
    cv::imshow("ErodeImage", erodeImage);
    cv::waitKey(0);
    cv::destroyWindow("ErodeImage");
    erodeImage.release();
    image.release();
    kernel.release();
}

int main(const int argc, const char * const argv[]) {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    lifuren::color("D:/tmp/logo.png");
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}

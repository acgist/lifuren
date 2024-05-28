#include "../../header/OpenCV.hpp"

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#include "Logger.hpp"

void lifuren::color(const std::string& path) {
    cv::Mat image = cv::imread(path);
    cv::Mat grayImage,
            blurImage,
            cannyImage,
            dilateImage,
            erodeImage;
    cv::imshow("Image", image);
    // 灰度
    cv::cvtColor(image, grayImage, cv::COLOR_RGB2GRAY);
    cv::imshow("GrayImage", grayImage);
    // 高斯模糊：size必须奇数
    cv::GaussianBlur(image, blurImage, cv::Size(3, 3), 3, 0);
    cv::imshow("BlurImage", blurImage);
    // 边缘检测
    cv::Canny(image, cannyImage, 16, 32);
    cv::imshow("CannyImage", cannyImage);
    // 图片膨胀
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(image, dilateImage, kernel);
    cv::imshow("DilateImage", dilateImage);
    // 图片侵蚀
    cv::erode(image, erodeImage, kernel);
    cv::imshow("ErodeImage", erodeImage);
    cv::waitKey(0);
    grayImage.release();
    blurImage.release();
    cannyImage.release();
    dilateImage.release();
    erodeImage.release();
    kernel.release();
    image.release();
}

int main(const int argc, const char * const argv[]) {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    lifuren::color("D:/tmp/logo.png");
    // lifuren::color("D:/tmp/Dota2.jpg");
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}

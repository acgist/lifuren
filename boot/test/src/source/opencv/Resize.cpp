#include "../../header/OpenCV.hpp"

void lifuren::resize(const std::string& path) {
    cv::Mat image = cv::imread(path);
    cv::Mat imageCrop;
    cv::Mat imageResize;
    cv::resize(image, imageResize, cv::Size(), 0.5, 0.5);
    cv::Rect crop(200, 200, 100, 100);
    imageCrop = image(crop);
    cv::imshow("Image", image);
    cv::imshow("ImageCrop", imageCrop);
    cv::imshow("ImageResize", imageResize);
    cv::waitKey(0);
    image.release();
    imageCrop.release();
    imageResize.release();
}

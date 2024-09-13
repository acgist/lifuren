/**
 * 图片工具
 * 
 * 图片读取、写入、变换、向量等等功能
 */
#ifndef LFR_HEADER_CV_IMAGES_HPP
#define LFR_HEADER_CV_IMAGES_HPP

#include <string>
#include <functional>

namespace cv {
    class Mat;
};

namespace lifuren {
namespace images  {

// 读取图片
extern bool read(const std::string& path, uint8_t** data, size_t& width, size_t& height, size_t& length);
// 写入图片
extern bool write(const std::string& path, uint8_t* data, size_t width, size_t height, size_t length = 0LL, size_t channel = 3LL);

/**
 * 读取图片
 * 
 * @param path      图片路径
 * @param data      图片数据
 * @param length    图片数据长度
 * @param width     目标图片宽度
 * @param height    目标图片高度
 * @param transform 图片变换
 */
extern void readTransform(
    const std::string& path,
    float * data,
    size_t& length,
    const int& width  = 0,
    const int& height = 0,
    const std::function<void(const ::cv::Mat&)> transform = nullptr
);

} // END OF images
} // END OF lifuren

#endif // END OF LFR_HEADER_CV_IMAGES_HPP
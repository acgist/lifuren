/**
 * 图片工具
 * 
 * 图片读取、写入、变换、向量等等功能
 */
#ifndef LFR_HEADER_CV_IMAGE_HPP
#define LFR_HEADER_CV_IMAGE_HPP

#include <string>
#include <functional>

namespace cv {
    class Mat;
};

namespace lifuren {
namespace image   {

/**
 * @param path   图片路径
 * @param data   图片数据
 * @param width  图片宽度
 * @param height 图片高度
 * @param length 数据长度
 * 
 * @return 是否成功
 */
extern bool read(const std::string& path, uint8_t** data, size_t& width, size_t& height, size_t& length);

/**
 * @param path    图片路径
 * @param data    图片数据
 * @param width   图片宽度
 * @param height  图片高度
 * @param length  数据长度
 * @param channel 通道数量
 * 
 * @return 是否成功
 */
extern bool write(const std::string& path, const uint8_t* data, const size_t& width, const size_t& height, size_t length = 0LL, const size_t& channel = 3LL);

/**
 * @param input         输入图片数据
 * @param width         输入图片宽度
 * @param height        输入图片高度
 * @param output        输出图片数据
 * @param output_width  输出图片宽度
 * @param output_height 输出图片高度
 * 
 * @return 是否成功
 */
extern bool resize(uint8_t* input, size_t width, size_t height, uint8_t* output, size_t output_width, size_t output_height);

/**
 * @param data   图片数据
 * @param width  图片宽度
 * @param height 图片高度
 * @param length 图片长度
 */
extern void show(const uint8_t* data, const size_t& width, const size_t& height, const size_t& length = 0LL);

/**
 * @param path      图片路径
 * @param data      图片数据
 * @param length    数据长度
 * @param width     目标宽度
 * @param height    目标高度
 * @param transform 图片变换
 */
extern void load(
    const std::string& path,
    float * data,
    size_t& length,
    const size_t& width  = 0LL,
    const size_t& height = 0LL,
    const std::function<void(const ::cv::Mat&)> transform = nullptr
);

} // END OF image
} // END OF lifuren

#endif // END OF LFR_HEADER_CV_IMAGE_HPP
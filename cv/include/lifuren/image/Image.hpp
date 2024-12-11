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
 * 
 * @return 是否成功
 */
extern bool read(const std::string& path,  char* data, const size_t& width, const size_t& height);
extern bool read(      cv::Mat    & image, char* data, const size_t& width, const size_t& height);

/**
 * @param path    图片路径
 * @param data    图片数据
 * @param width   图片宽度
 * @param height  图片高度
 * 
 * @return 是否成功
 */
extern bool write(const std::string& path, const char* data, const size_t& width, const size_t& height);

} // END OF image
} // END OF lifuren

#endif // END OF LFR_HEADER_CV_IMAGE_HPP
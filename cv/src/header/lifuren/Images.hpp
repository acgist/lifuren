/**
 * 图片工具
 * 
 * 图片读取、写入、变换、向量等等功能
 */
#ifndef LFR_HEADER_CV_IMAGES_HPP
#define LFR_HEADER_CV_IMAGES_HPP

#include <string>

namespace lifuren {
namespace images  {

// 读取图片
extern bool read(const std::string& path, uint8_t** data, size_t& width, size_t& height, size_t& length);
// 写入图片
extern bool write(const std::string& path, uint8_t* data, size_t width, size_t height, size_t length = 0LL, size_t channel = 3LL);

} // END OF images
} // END OF lifuren

#endif // END OF LFR_HEADER_CV_IMAGES_HPP
/**
 * CV
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CV_CV_HPP
#define LFR_HEADER_CV_CV_HPP

#include <cstdint>
#include <cstdlib>

namespace lifuren {
namespace cv      {

/**
 * 配置OpenCV日志
 */
extern void logger();

extern void show(uint8_t* data, size_t width, size_t height, size_t length);

} // END OF cv
} // END OF lifuren

#endif // END OF LFR_HEADER_CV_CV_HPP

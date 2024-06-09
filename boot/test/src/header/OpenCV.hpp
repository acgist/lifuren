/**
 * OpenCV
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_BOOT_OPENCV_HPP
#define LFR_HEADER_BOOT_OPENCV_HPP

#include <string>

namespace lifuren {

/**
 * 人脸识别
 * 
 * @param model 模型路径
 * @param path  图片路径
 */
extern void face(const std::string& model, const std::string& path);

/**
 * 修改图片颜色
 * 
 * @param path 图片路径
 */
extern void color(const std::string& path);

/**
 * 修改图片大小
 * 
 * @param path 图片路径
 */
extern void resize(const std::string& path);
    
}

#endif // LFR_HEADER_BOOT_OPENCV_HPP

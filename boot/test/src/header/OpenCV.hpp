/**
 * OpenCV
 * 
 * @author acgist
 */
#pragma once

#include <string>

#include "Logger.hpp"

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/core/utils/logger.hpp"

namespace lifuren {

/**
 * 关闭日志
 */
extern void shutdownOpenCVLogger();
/**
 * 图片读取和显示
 * 
 * @param path 图片路径
 */
extern void readAndShow(const std::string& path);
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
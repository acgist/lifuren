/**
 * OpenCV
 * 
 * @author acgist
 */
#pragma once

#include <string>

#include "GLog.hpp"

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/core/utils/logger.hpp"

namespace lifuren {

extern void shutdownOpenCVLogger();
extern void readAndShow(const std::string& path);
extern void color(const std::string& path);
extern void resize(const std::string& path);
    
}
#pragma once

#include <string>
#include <vector>

#include "mgl2/mgl.h"
#include "dlib/svm.h"
#include "opencv2/opencv.hpp"

namespace lifuren {

namespace gg {

    /**
     * 绘制散点图
     */
    extern void dots(std::vector<double>* x, std::vector<double>* y, int length, int xLength = 100, int yLength = 100, std::vector<double>* xd = nullptr, std::vector<double>* yd = nullptr, const char* title = "lifuren", int width = 800, int height = 800);

}

}
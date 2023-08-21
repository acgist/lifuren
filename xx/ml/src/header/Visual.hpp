#pragma once

#include <string>
#include <vector>

#include "mgl2/mgl.h"
#include "dlib/svm.h"
#include "opencv2/opencv.hpp"

namespace lifuren {

    /**
     * 绘制散点图
     * 
     * @param x       X轴数据
     * @param y       Y轴数据
     * @param length  数据长度
     * @param xLength X轴长度
     * @param yLength Y轴长度
     * @param xd      X轴对比数据
     * @param yd      Y轴对比数据
     * @param title   标题
     * @param width   画布宽度
     * @param height  画布高度
     */
    extern void dots(std::vector<double>* x, std::vector<double>* y, int length, int xLength = 100, int yLength = 100, std::vector<double>* xd = nullptr, std::vector<double>* yd = nullptr, const char* title = "lifuren", int width = 800, int height = 800);

}
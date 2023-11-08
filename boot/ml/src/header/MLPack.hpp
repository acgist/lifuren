#pragma once

#include "GLog.hpp"
#if defined(__unix__) || defined(__linux__)
#include "mlpack.hpp"
#elif defined(_WIN32)
#include "mlpack/mlpack.hpp"
#else
#error "不支持的操作系统"
#endif

namespace lifuren {

/**
 * 矩阵
 */
extern void testMLPackMatrix();

/**
 * 加载数据
 * 
 * @param path 文件路径
 */
extern void testMLPackLoadFile(const char* path);

/**
 * 线性回归
 */
extern void testMLPackLinearRegression();

/**
 * Softmax逻辑回归（多分类问题）
 */
extern void testMLPackSoftmaxRegression();

/**
 * 逻辑回归（二分类问题）
 */
extern void testMLPackLogisticRegression();

}

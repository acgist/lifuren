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
 * 加载数据
 */
extern void testLoadFile();

/**
 * 矩阵
 */
extern void testMLPackMatrix();

/**
 * 线性回归
 */
extern void testMLPackLinearRegression();

/**
 * 逻辑回归（二分类问题）
 */
extern void testMLPackLogisticRegression();

/**
 * Softmax逻辑回归（多分类问题）
 */
extern void testMLPackSoftmaxRegression();

}

#pragma once

#include "GLog.hpp"
#ifdef _WIN32
#include "mlpack/mlpack.hpp"
#else
#include "mlpack.hpp"
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
 * 逻辑回归
 */
extern void testMLPackLogisticRegression();

}

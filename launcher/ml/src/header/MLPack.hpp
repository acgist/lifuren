#pragma once

#include "GLog.hpp"
#ifdef _WIN32
#include "mlpack/mlpack.hpp"
#else
#include "mlpack.hpp"
#endif

namespace lifuren {

/**
 * 矩阵
 */
extern void matrix();

/**
 * 线性回归
 */
extern void linearRegression();

}
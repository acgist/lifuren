/**
 * MLPack
 * 
 * @author acgist
 */
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

extern void testMLPackMatrix();
extern void testMLPackLoadFile(const char* path);
extern void testMLPackLinearRegression();
extern void testMLPackSoftmaxRegression();
extern void testMLPackLogisticRegression();

}

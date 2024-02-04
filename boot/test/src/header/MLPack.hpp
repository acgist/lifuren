/**
 * MLPack
 * 
 * @author acgist
 */
#pragma once

#include "GLog.hpp"

// 不同系统位置不同
#if defined(__unix__) || defined(__linux__)
#include "mlpack.hpp"
#elif defined(_WIN32)
#include "mlpack/mlpack.hpp"
#else
#error "不支持的操作系统"
#endif

namespace lifuren {

/**
 * MLPack Matrix测试
 */
extern void testMLPackMatrix();
/**
 * MLPack加载数据文件测试
 * 
 * @param path 数据文件路径
 */
extern void testMLPackLoadFile(const char* path);
/**
 * MLPack LinearRegression测试
 */
extern void testMLPackLinearRegression();
/**
 * MLPack SoftmaxRegression测试
 */
extern void testMLPackSoftmaxRegression();
/**
 * MLPack LogisticRegression测试
 */
extern void testMLPackLogisticRegression();

}

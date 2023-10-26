#pragma once

#include "GLog.hpp"
#include "torch/torch.h"

namespace lifuren {

extern void testMatrix();

extern void testReLU();

extern void testTanh();

/**
 * https://blog.csdn.net/m0_59158839/article/details/126813648
 */
extern void testLinearRegression();

extern void testEmbedding();

}
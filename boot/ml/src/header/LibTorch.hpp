#pragma once

#include "GLog.hpp"
#include "torch/torch.h"

namespace lifuren {

extern void testMatrix();

extern void testReLU();

extern void testTanh();

extern void testLinearRegression();

extern void testEmbedding();

/**
 * 风格迁移
 */
extern void testTS();

/**
 * VGG
 */
extern void testVGG();

/**
 * LSTM
 */
extern void testLSTM();

/**
 * DCGAN
 */
extern void testDCGAN();

}
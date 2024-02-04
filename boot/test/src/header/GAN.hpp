/**
 * GAN
 * 
 * @author acgist
 */
#pragma once

#include "torch/torch.h"

namespace lifuren {

/**
 * GAN测试
 */
extern void testGAN();
/**
 * DCGAN测试
 */
extern void testDCGAN();
/**
 * CycleGAN测试
 * https://blog.csdn.net/jizhidexiaoming/article/details/128619117
 */
extern void testCycleGAN();
/**
 * StyleGAN测试
 */
extern void testStyleGAN();

}
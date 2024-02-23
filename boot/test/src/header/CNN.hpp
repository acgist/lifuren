/**
 * CNN
 * 
 * @author acgist
 */
#pragma once

#include "GLog.hpp"

#include "torch/torch.h"

namespace lifuren {

/**
 * VGG测试
 */
extern void testVGG();
/**
 * MNIST测试
 */
extern void testMNIST();
/**
 * ResNet测试
 */
extern void testResNet();

}
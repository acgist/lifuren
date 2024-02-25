/**
 * CNN
 * 
 * @author acgist
 */
#pragma once

#include "Logger.hpp"

#include "torch/torch.h"

namespace lifuren {

/**
 * VGG测试
 */
extern void testVGG();
/**
 * MNIST测试
 */
extern void testMnist();
/**
 * ResNet测试
 */
extern void testResNet();

}
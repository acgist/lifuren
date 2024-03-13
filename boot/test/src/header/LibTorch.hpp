/**
 * LibTorch
 * 
 * @author acgist
 */
#pragma once

#include "Logger.hpp"

#include "torch/torch.h"

LFR_LOG_FORMAT(at::Tensor);

namespace lifuren {

/**
 * 是否支持CUDA
 */
extern void testCUDA();
/**
 * LibTorch Tensor测试
 */
extern void testLibTorchTensor();

}
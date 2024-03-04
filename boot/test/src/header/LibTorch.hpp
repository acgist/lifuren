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
 * LibTorch ReLU测试
 */
extern void testLibTorchReLU();
/**
 * LibTorch Tanh测试
 */
extern void testLibTorchTanh();
/**
 * LibTorch Tensor测试
 */
extern void testLibTorchTensor();

}
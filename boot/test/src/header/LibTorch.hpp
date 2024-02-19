/**
 * LibTorch
 * 
 * @author acgist
 */
#pragma once

#include "GLog.hpp"

#include "torch/torch.h"

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
/**
 * LibTorch LinearRegression测试
 */
extern void testLibTorchLinearRegression();

}
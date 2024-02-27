/**
 * LibTorch
 * 
 * @author acgist
 */
#pragma once

#include "Logger.hpp"

#include "torch/torch.h"

#include "spdlog/fmt/ostr.h"

/**
 * Tensor格式化
 */
template <> struct fmt::formatter<at::Tensor> : ostream_formatter {};

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
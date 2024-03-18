/**
 * LibTorch
 * 
 * @author acgist
 */
#pragma once

#include "torch/torch.h"

#include "Logger.hpp"

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

/**
 * 性别识别
 */
class GenderRecognition {

public:
    void trian();
    void test();
    void pred();

};

}
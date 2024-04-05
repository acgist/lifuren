/**
 * LibTorch
 * 
 * @author acgist
 */
#pragma once

#include "./LibTorchCV.hpp"
#include "./LibTorchNLP.hpp"

LFR_LOG_FORMAT(at::Tensor);

namespace lifuren {

/**
 * 张量测试
 */
extern void testLibTorchTensor();

/**
 * 目标检测
 */
extern void testYOLO();

/**
 * 语义分割
 */
extern void testDeepLab();

}

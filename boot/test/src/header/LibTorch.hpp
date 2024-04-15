/**
 * LibTorch
 * 
 * @author acgist
 */
#pragma once

#include <vector>

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/core/utils/logger.hpp"

#include "./LibTorchCV.hpp"
#include "./LibTorchNLP.hpp"

LFR_LOG_FORMAT_STREAM(at::Tensor);

namespace lifuren {

/**
 * 张量测试
 */
extern void testLibTorchTensor();

/**
 * 模型测试
 */
extern void testModel();

/**
 * 目标检测
 */
extern void testYOLO();

/**
 * 语义分割
 */
extern void testDeepLab();

}

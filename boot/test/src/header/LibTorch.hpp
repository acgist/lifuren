/**
 * LibTorch
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_BOOT_LIBTORCH_HPP
#define LFR_HEADER_BOOT_LIBTORCH_HPP

#include <string>
#include <vector>

#include "torch/torch.h"
#include "torch/script.h"

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/core/utils/logger.hpp"

#include "utils/Layers.hpp"
#include "config/Config.hpp"
#include "utils/Datasets.hpp"

#include "spdlog/fmt/ostr.h"
#include "spdlog/fmt/chrono.h"
#include "spdlog/fmt/ranges.h"

LFR_LOG_FORMAT_STREAM(at::Tensor);
LFR_LOG_FORMAT_STREAM(torch::jit::IValue);

namespace lifuren {

/**
 * 模型测试
 */
extern void testModel();

/**
 * 张量测试
 */
extern void testTensor();

}

#endif // LFR_HEADER_BOOT_LIBTORCH_HPP

/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * Torch
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_CORE_TORCH_HPP
#define LFR_HEADER_CORE_TORCH_HPP

#include <string>

#include "torch/types.h"

namespace lifuren {

/**
 * @return 设备类型
 */
extern torch::DeviceType getDevice();

/**
 * 记录日志
 */
extern void logTensor(
    const std::string& message, // 日志
    const at::Tensor & tensor   // 张量
);

/**
 * 记录日志
 */
extern void logTensor(
    const std::string     & message, // 日志
    const c10::IntArrayRef& tensor   // 张量
);

} // END OF lifuren

#endif // END OF LFR_HEADER_CORE_TORCH_HPP

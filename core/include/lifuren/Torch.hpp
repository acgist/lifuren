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
extern torch::DeviceType get_device();

/**
 * @param message 日志
 * @param tensor  张量
 */
extern void log_tensor(const std::string& message, const at::Tensor& tensor);

/**
 * @param message 日志
 * @param tensor  张量
 */
extern void log_tensor(const std::string& message, const c10::IntArrayRef& tensor);

} // END OF lifuren

#endif // END OF LFR_HEADER_CORE_TORCH_HPP

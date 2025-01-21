/**
 * Torch
 */
#ifndef LFR_HEADER_MODEL_TORCH_HPP
#define LFR_HEADER_MODEL_TORCH_HPP

#include <string>

#include "torch/types.h"

namespace lifuren {

extern void setDevice(torch::DeviceType& type);

extern void logTensor(const std::string& message, const at::Tensor& tensor);

extern void logTensor(const std::string& message, const c10::IntArrayRef& tensor);

} // END OF lifuren

#endif // END OF LFR_HEADER_MODEL_TORCH_HPP

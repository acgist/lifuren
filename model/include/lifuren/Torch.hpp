/**
 * Torch
 */
#ifndef LFR_HEADER_MODEL_TORCH_HPP
#define LFR_HEADER_MODEL_TORCH_HPP

#include "torch/types.h"

namespace lifuren {

extern void setDevice(torch::DeviceType& type);

extern void logTensor(const torch::Tensor& tensor);

extern void logTensor(const c10::IntArrayRef& tensor);

extern void quantization(const std::string& model_path);

} // END OF lifuren

#endif // END OF LFR_HEADER_MODEL_TORCH_HPP

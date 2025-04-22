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
#include <vector>
#include <fstream>

#include "torch/types.h"

namespace lifuren {

/**
 * @return 设备类型
 */
extern torch::DeviceType getDevice();

/**
 * @param message 日志
 * @param tensor  张量
 */
extern void logTensor(const std::string& message, const at::Tensor& tensor);

/**
 * @param message 日志
 * @param tensor  张量
 */
extern void logTensor(const std::string& message, const c10::IntArrayRef& tensor);

/**
 * @param stream 文件流
 */
inline at::Tensor read_tensor(std::ifstream& stream) {
    size_t size = 0;
    if(!stream.read(reinterpret_cast<char*>(&size), sizeof(size_t))) {
        return {};
    }
    std::vector<int64_t> vector(size);
    stream.read(reinterpret_cast<char*>(vector.data()), sizeof(int64_t) * size);
    c10::IntArrayRef sizes(vector);
    torch::Tensor tensor = torch::zeros(sizes, torch::kFloat32);
    stream.read(reinterpret_cast<char*>(tensor.data_ptr()), tensor.numel() * tensor.element_size());
    return tensor;
}

/**
 * @param stream 文件流
 * @param tensor 张量
 */
inline void write_tensor(std::ofstream& stream, const at::Tensor& tensor) {
    const auto sizes = tensor.sizes();
    const auto size  = sizes.size();
    stream.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
    stream.write(reinterpret_cast<const char*>(sizes.data()), sizeof(int64_t) * size);
    stream.write(reinterpret_cast<const char*>(tensor.const_data_ptr()), tensor.numel() * tensor.element_size());
}

} // END OF lifuren

#endif // END OF LFR_HEADER_CORE_TORCH_HPP

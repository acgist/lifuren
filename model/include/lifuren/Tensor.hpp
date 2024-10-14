/**
 * 矩阵工具
 */
#ifndef LFR_HEADER_MODEL_TENSOR_HPP
#define LFR_HEADER_MODEL_TENSOR_HPP

#include "ggml.h"

#include <random>

namespace lifuren {
namespace tensor  {

/**
 * @param tensor 张量
 * @param value  填充
 */
inline void fill(ggml_tensor* tensor, float value = 0.0F) {
    GGML_ASSERT(tensor->type == GGML_TYPE_F32);
    int64_t size = ggml_nelements(tensor);
    float * data = ggml_get_data_f32(tensor);
    std::fill(data, data + size, value);
}

/**
 * @param tensor 张量
 * @param value  填充
 */
inline void fill(ggml_tensor* tensor, const float* value) {
    GGML_ASSERT(tensor->type == GGML_TYPE_F32);
    int64_t size = ggml_nelements(tensor);
    float * data = ggml_get_data_f32(tensor);
    std::copy_n(value, size, data);
}

/**
 * @param tensor 张量
 * @param mean   均值
 * @param sigma  方差
 */
inline void fillRand(ggml_tensor* tensor, float mean = 0.0F, float sigma = 0.001F) {
    GGML_ASSERT(tensor->type == GGML_TYPE_F32);
    std::random_device device{};
    std::mt19937 random{device()};
    std::normal_distribution<float> normal(mean, sigma);
    // std::bind(normal, random);
    int64_t size = ggml_nelements(tensor);
    float * data = ggml_get_data_f32(tensor);
    for (int64_t i = 0; i < size; ++i) {
        data[i] = normal(random);
    }
}

/**
 * @param tensor 张量
 * @param begin  开始
 * @param delta  增量
 */
inline void fillRange(ggml_tensor* tensor, float begin = 0.0F, const float delta = 1.0F) {
    GGML_ASSERT(tensor->type == GGML_TYPE_F32);
    int64_t size = ggml_nelements(tensor);
    float * data = ggml_get_data_f32(tensor);
    for(int i = 0; i < size; ++i) {
        data[i] = begin;
        begin  += delta;
    }
}

/**
 * 打印张量信息
 * 
 * @param tensor  张量
 * @param log     日志输出
 */
extern std::string print(const ggml_tensor* tensor, const bool log = true);

} // END OF tensor
} // END OF lifuren

#endif // END OF LFR_HEADER_MODEL_TENSOR_HPP

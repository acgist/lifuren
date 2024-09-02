/**
 * 矩阵
 */
#ifndef LFR_HEADER_MODEL_TENSORS_HPP
#define LFR_HEADER_MODEL_TENSORS_HPP

#include "ggml.h"

#include <random>

namespace lifuren {
namespace tensors {

inline void fill(ggml_tensor* tensor, float value = 0.0F) {
    GGML_ASSERT(tensor->type == GGML_TYPE_F32);
    int64_t size = ggml_nelements(tensor);
    float * data = ggml_get_data_f32(tensor);
    for(int i = 0; i < size; ++i) {
        data[i] = value;
    }
}

inline void fillRand(ggml_tensor* tensor, float mean = 0.0F, float sigma = 0.001F) {
    GGML_ASSERT(tensor->type == GGML_TYPE_F32);
    std::random_device device{};
    std::mt19937 random{device()};
    std::normal_distribution<float> normal(mean, sigma);
    int64_t size = ggml_nelements(tensor);
    float * data = ggml_get_data_f32(tensor);
    for (int64_t i = 0; i < size; ++i) {
        data[i] = normal(random);
    }
}

inline void fillRange(ggml_tensor* tensor, float value = 0.0F) {
    GGML_ASSERT(tensor->type == GGML_TYPE_F32);
    int64_t size = ggml_nelements(tensor);
    float * data = ggml_get_data_f32(tensor);
    for(int i = 0; i < size; ++i) {
        data[i] = value + i;
    }
}

} // END OF tensors
} // END OF lifuren

#endif // END OF LFR_HEADER_MODEL_TENSORS_HPP

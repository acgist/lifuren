/**
 * layer工具
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_MODEL_LAYERS_HPP
#define LFR_HEADER_MODEL_LAYERS_HPP

#include <vector>

#include "ggml.h"

namespace lifuren {
namespace layers  {

class Layer {

public:
    Layer();
    virtual ~Layer();

public:
    virtual ggml_tensor* forward(ggml_tensor* input);
    virtual ggml_tensor* operator()(ggml_tensor* input);


};

/**
 * @param in_features  输入特征大小
 * @param out_features 输出特征大小
 * 
 * @return Linear
 */
inline void linear(int64_t in_features, int64_t out_features) {
    // TODO
}

/**
 * @param num_features 特征大小
 * 
 * @return BatchNorm1d
 */
inline void batchNorm1d(int64_t num_features) {
    // TODO
}

/**
 * @param num_features 特征大小
 * 
 * @return BatchNorm2d
 */
inline void batchNorm2d(int64_t num_features) {
    // TODO
}

/**
 * @param normalized_shape TODO: 学习
 * 
 * @return LayerNorm
 */
inline void layerNorm(std::vector<int64_t> normalized_shape) {
    // TODO
}

/**
 * @param num_features TODO: 学习
 * 
 * @return InstanceNorm2d
 */
inline void instanceNorm2d(int64_t num_features) {
    // TODO
}

/**
 * @param num_groups   TODO: 学习
 * @param num_channels TODO: 学习
 * 
 * @return GroupNorm
 */
inline void groupNorm(int64_t num_groups, int64_t num_channels) {
    // TODO
}

/**
 * @param in_channels  输入通道大小
 * @param out_channels 输出通道大小
 * @param kernel_size  卷积核大小
 * @param stride       步长
 * @param padding      填充
 * @param dilation     间隔
 * @param bias         偏置
 * 
 * @return Conv2d
 */
inline void conv2d(
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    int64_t stride   = 1,
    int64_t padding  = 0,
    int64_t dilation = 1,
    bool    bias     = true
) {
    // TODO
}

/**
 * @param kernel_size 卷积核大小
 * @param stride      步长
 * @param padding     填充
 * @param dilation    间隔
 * 
 * @return MaxPool2d
 */
inline void maxPool2d(
    int64_t kernel_size,
    int64_t stride   = -1,
    int64_t padding  = 0,
    int64_t dilation = 1
) {
    // TODO
}

/**
 * @param kernel_size 卷积核大小
 * @param stride      步长
 * @param padding     填充
 * 
 * @return MaxPool2d
 */
inline void avgPool2d(
    int64_t kernel_size,
    int64_t stride  = -1,
    int64_t padding = 0
) {
    // TODO
}

/**
 * @param output_size 输出尺寸
 * 
 * @return AdaptiveAvgPool2d
 */
inline void adaptiveAvgPool2d(int64_t output_size) {
    // TODO
}

/**
 * @param dropout 丢弃概率
 * 
 * @return Dropout
 */
inline void dropout(double dropout = 0.5) {
    // TODO
}

/**
 * @param inplace 是否使用旧的内存
 * 
 * @return ReLU
 */
inline void relu(bool inplace = false) {
    // TODO
}

/**
 * @param input_size  ?
 * @param hidden_size ?
 * @param num_layers  ?
 * @param bias        ?
 * @param dropout     ?
 * 
 * @return GRU
 */
inline void gru(
    int64_t input_size,
    int64_t hidden_size,
    int64_t num_layers = 1,
    bool    bias       = true,
    double  dropout    = 0.0
) {
    // TODO
}

/**
 * @param input_size  ?
 * @param hidden_size ?
 * @param num_layers  ?
 * @param bias        ?
 * @param dropout     ?
 * 
 * @return LSTM
 */
inline void lstm(
    int64_t input_size,
    int64_t hidden_size,
    int64_t num_layers = 1,
    bool    bias       = true,
    double  dropout    = 0.0
) {
    // TODO
}

} // END layers
} // END lifuren

#endif // LFR_HEADER_MODEL_LAYERS_HPP

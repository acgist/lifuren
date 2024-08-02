/**
 * layer工具
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_MODEL_LAYERS_HPP
#define LFR_HEADER_MODEL_LAYERS_HPP

#include "torch/torch.h"

namespace lifuren {
namespace layers  {

/**
 * @param in_features  输入特征大小
 * @param out_features 输出特征大小
 * 
 * @return Linear
 */
inline torch::nn::Linear linear(int64_t in_features, int64_t out_features) {
    return torch::nn::Linear(torch::nn::LinearOptions(in_features, out_features));
}

/**
 * @param num_features 特征大小
 * 
 * @return BatchNorm1d
 */
inline torch::nn::BatchNorm1d batchNorm1d(int64_t num_features) {
    return torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(num_features));
}

/**
 * @param num_features 特征大小
 * 
 * @return BatchNorm2d
 */
inline torch::nn::BatchNorm2d batchNorm2d(int64_t num_features) {
    return torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(num_features));
}

/**
 * @param normalized_shape TODO: 学习
 * 
 * @return LayerNorm
 */
inline torch::nn::LayerNorm layerNorm(std::vector<int64_t> normalized_shape) {
    return torch::nn::LayerNorm(torch::nn::LayerNormOptions(normalized_shape));
}

/**
 * @param num_features TODO: 学习
 * 
 * @return InstanceNorm2d
 */
inline torch::nn::InstanceNorm2d instanceNorm2d(int64_t num_features) {
    return torch::nn::InstanceNorm2d(torch::nn::InstanceNorm2dOptions(num_features));
}

/**
 * @param num_groups   TODO: 学习
 * @param num_channels TODO: 学习
 * 
 * @return GroupNorm
 */
inline torch::nn::GroupNorm groupNorm(int64_t num_groups, int64_t num_channels) {
    return torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, num_channels));
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
inline torch::nn::Conv2d conv2d(
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    int64_t stride   = 1,
    int64_t padding  = 0,
    int64_t dilation = 1,
    bool    bias     = true
) {
    torch::nn::Conv2dOptions options(in_channels, out_channels, kernel_size);
    options.stride(stride);
    options.padding(padding);
    options.dilation(dilation);
    options.bias(bias);
    return torch::nn::Conv2d(options);
}

/**
 * @param kernel_size 卷积核大小
 * @param stride      步长
 * @param padding     填充
 * @param dilation    间隔
 * 
 * @return MaxPool2d
 */
inline torch::nn::MaxPool2d maxPool2d(
    int64_t kernel_size,
    int64_t stride   = -1,
    int64_t padding  = 0,
    int64_t dilation = 1
) {
    torch::nn::MaxPool2dOptions options(kernel_size);
    options.stride(stride < 0 ? kernel_size : stride);
    options.padding(padding);
    options.dilation(dilation);
    return torch::nn::MaxPool2d(options);
}

/**
 * @param kernel_size 卷积核大小
 * @param stride      步长
 * @param padding     填充
 * 
 * @return MaxPool2d
 */
inline torch::nn::AvgPool2d avgPool2d(
    int64_t kernel_size,
    int64_t stride  = -1,
    int64_t padding = 0
) {
    torch::nn::AvgPool2dOptions options(kernel_size);
    options.stride(stride < 0 ? kernel_size : stride);
    options.padding(padding);
    return torch::nn::AvgPool2d(options);
}

/**
 * @param output_size 输出尺寸
 * 
 * @return AdaptiveAvgPool2d
 */
inline torch::nn::AdaptiveAvgPool2d adaptiveAvgPool2d(int64_t output_size) {
    return torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(output_size));
}

/**
 * @param dropout 丢弃概率
 * 
 * @return Dropout
 */
inline torch::nn::Dropout dropout(double dropout = 0.5) {
    return torch::nn::Dropout(torch::nn::DropoutOptions(dropout));
}

/**
 * @param inplace 是否使用旧的内存
 * 
 * @return ReLU
 */
inline torch::nn::ReLU relu(bool inplace = false) {
    return torch::nn::ReLU(torch::nn::ReLUOptions(inplace));
}

// /**
//  * @return torch::nn::Upsample
//  */
// inline torch::nn::Upsample upsample() {
//     torch::nn::UpsampleOptions options();
//     return torch::nn::Upsample(options);
// }

/**
 * @param input_size  ?
 * @param hidden_size ?
 * @param num_layers  ?
 * @param bias        ?
 * @param dropout     ?
 * 
 * @return torch::nn::LSTM
 */
inline torch::nn::GRU gru(
    int64_t input_size,
    int64_t hidden_size,
    int64_t num_layers = 1,
    bool    bias       = true,
    double  dropout    = 0.0
) {
    torch::nn::GRUOptions options(input_size, hidden_size);
    options.num_layers(num_layers);
    options.bias(bias);
    options.dropout(dropout);
    return torch::nn::GRU(options);
}

/**
 * @param input_size  ?
 * @param hidden_size ?
 * @param num_layers  ?
 * @param bias        ?
 * @param dropout     ?
 * 
 * @return torch::nn::LSTM
 */
inline torch::nn::LSTM lstm(
    int64_t input_size,
    int64_t hidden_size,
    int64_t num_layers = 1,
    bool    bias       = true,
    double  dropout    = 0.0
) {
    torch::nn::LSTMOptions options(input_size, hidden_size);
    options.num_layers(num_layers);
    options.bias(bias);
    options.dropout(dropout);
    return torch::nn::LSTM(options);
}

} // END layers
} // END lifuren

#endif // LFR_HEADER_MODEL_LAYERS_HPP

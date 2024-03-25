/**
 * layer工具
 * 
 * @author acgist
 */
#pragma once

#include "torch/torch.h"

namespace lifuren {
namespace layers  {

/**
 * @param in_features  输入特征大小
 * @param out_features 输出特征大小
 */
inline torch::nn::LinearOptions linearOptions(int64_t in_features, int64_t out_features) {
    return torch::nn::LinearOptions(in_features, out_features);
}

/**
 * 线性变换
 * 
 * @param in_features  输入特征大小
 * @param out_features 输出特征大小
 */
inline torch::nn::Linear linear(int64_t in_features, int64_t out_features) {
    return torch::nn::Linear(linearOptions(in_features, out_features));
}

/**
 * @param num_features 特征大小
 */
inline torch::nn::BatchNorm1dOptions batchNorm1dOptions(int64_t num_features) {
    return torch::nn::BatchNorm1dOptions(num_features);
}

/**
 * @param num_features 特征大小
 */
inline torch::nn::BatchNorm1d batchNorm1d(int64_t num_features) {
    return torch::nn::BatchNorm1d(batchNorm1dOptions(num_features));
}

/**
 * @param num_features 特征大小
 */
inline torch::nn::BatchNorm2dOptions batchNorm2dOptions(int64_t num_features) {
    return torch::nn::BatchNorm2dOptions(num_features);
}

/**
 * @param num_features 特征大小
 */
inline torch::nn::BatchNorm2d batchNorm2d(int64_t num_features) {
    return torch::nn::BatchNorm2d(batchNorm2dOptions(num_features));
}

/**
 * @param in_channels  输入通道大小
 * @param out_channels 输出通道大小
 * @param kernel_size  卷积核大小
 * @param stride       步长
 * @param padding      填充
 * @param bias         偏置
 */
inline torch::nn::Conv2dOptions conv2dOptions(
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    int64_t stride  = 1,
    int64_t padding = 0,
    bool    bias    = false
) {
    torch::nn::Conv2dOptions options = torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size);
    options.stride(stride);
    options.padding(padding);
    options.bias(bias);
    return options;
}

/**
 * 卷积
 * 
 * @param in_channels  输入通道大小
 * @param out_channels 输出通道大小
 * @param kernel_size  卷积核大小
 * @param stride       步长
 * @param padding      填充
 * @param bias         偏置
 */
inline torch::nn::Conv2d conv2d(
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    int64_t stride  = 1,
    int64_t padding = 0,
    bool    bias    = false
) {
    return torch::nn::Conv2d(conv2dOptions(in_channels, out_channels, kernel_size, stride, padding, bias));
}

// inline torch::nn::MaxPool2dOptions maxPool2dOptions() {

// }

// MaxPool2d
// AdaptiveAvgPool2d
// Conv2dReluBN
// torch::nn::LSTM

}
}
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
 * 线性变换
 * 
 * @param in_features  输入特征大小
 * @param out_features 输出特征大小
 */
inline torch::nn::Linear linear(int64_t in_features, int64_t out_features) {
    return torch::nn::Linear(torch::nn::LinearOptions(in_features, out_features));
}

/**
 * @param num_features 特征大小
 */
inline torch::nn::BatchNorm1d batchNorm1d(int64_t num_features) {
    return torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(num_features));
}

/**
 * @param num_features 特征大小
 */
inline torch::nn::BatchNorm2d batchNorm2d(int64_t num_features) {
    return torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(num_features));
}

/**
 * 卷积
 * 
 * @param in_channels  输入通道大小
 * @param out_channels 输出通道大小
 * @param kernel_size  卷积核大小
 * @param stride       步长
 * @param padding      填充
 * @param dilation     元素间隔
 * @param bias         偏置
 */
inline torch::nn::Conv2d conv2d(
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    int64_t stride   = 1,
    int64_t padding  = 0,
    int64_t dilation = 1,
    bool    bias     = false
) {
    torch::nn::Conv2dOptions options(in_channels, out_channels, kernel_size);
    options.stride(stride);
    options.padding(padding);
    options.dilation(dilation);
    options.bias(bias);
    return torch::nn::Conv2d(options);
}

/**
 * 最大池化
 * 
 * TODO: stride默认值的大小
 * 
 * @param kernel_size 卷积核大小
 * @param stride      步长
 * @param padding     填充
 * @param dilation    元素间隔
 */
inline torch::nn::MaxPool2d maxPool2d(
    int64_t kernel_size,
    int64_t stride   = 1,
    // int64_t stride   = kernel_size,
    int64_t padding  = 0,
    int64_t dilation = 1
) {
    torch::nn::MaxPool2dOptions options(kernel_size);
    options.stride(stride);
    options.padding(padding);
    options.dilation(dilation);
    return torch::nn::MaxPool2d(options);
}

inline torch::nn::AvgPool2d avgPool2d() {

}

inline torch::nn::AdaptiveAvgPool2d adaptiveAvgPool2d() {
}

inline void mlp() {
}

inline torch::nn::LSTM lstm() {
}

inline void conv2dLayer() {
}

}
}
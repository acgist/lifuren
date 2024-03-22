/**
 * LibTorch
 * 
 * @author acgist
 */
#pragma once

#include <string>

#include "torch/torch.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#include "Logger.hpp"
#include "utils/Datasets.hpp"

LFR_LOG_FORMAT(at::Tensor);

namespace lifuren {

/**
 * LibTorch Tensor测试
 */
extern void testLibTorchTensor();

/**
 * 性别模型（参考VGG）
 * 设置不要计算梯度：层.value().set_requires_grad()
 */
class GenderImpl : public torch::nn::Module {

private:
    // 卷积层
    torch::nn::Sequential        features{ nullptr };
    // 池化层
    torch::nn::AdaptiveAvgPool2d avgPool{ nullptr };
    // 全连接层
    torch::nn::Sequential        classifier;

public:
    GenderImpl(std::vector<int>& cfg, int num_classes = 1000, bool batch_norm = false);
    torch::Tensor forward(torch::Tensor x);

};
TORCH_MODULE(Gender);

/**
 * Conv2dOptions
 * 二维卷积参数配置
 * 
 * @param in_planes
 * @param out_planes
 * @param kerner_size
 */
inline torch::nn::Conv2dOptions conv2dOptions(
    int64_t in_planes,
    int64_t out_planes,
    int64_t kerner_size,
    int64_t stride = 1,
    int64_t padding = 0,
    bool with_bias = false
) {
    torch::nn::Conv2dOptions options = torch::nn::Conv2dOptions(in_planes, out_planes, kerner_size);
    options.stride(stride);
    options.padding(padding);
    options.bias(with_bias);
    return options;
}

/**
 * MaxPool2dOptions
 * 最大池化参数配置
 * 
 * @param kernel_size
 * @param stride
 */
inline torch::nn::MaxPool2dOptions maxPool2dOptions(int kernel_size, int stride){
    torch::nn::MaxPool2dOptions options(kernel_size);
    options.stride(stride);
    return options;
}

/**
 * 特征提取层
 * 
 * @param cfg
 * @param batch_norm 归一化
 */
inline torch::nn::Sequential makeFeatures(std::vector<int>& cfg, bool batch_norm) {
    int in_channels = 3;
    torch::nn::Sequential features;
    for (auto v : cfg) {
        if (v == -1) {
            features->push_back(torch::nn::MaxPool2d(lifuren::maxPool2dOptions(2, 2)));
        } else {
            features->push_back(torch::nn::Conv2d(lifuren::conv2dOptions(in_channels, v, 3, 1, 1)));
            if (batch_norm) {
                features->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(v)));
            }
            features->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(true)));
            in_channels = v;
        }
    }
    return features;
}

/**
 * 性别识别
 */
class GenderHandler {

public:
    torch::Device device = torch::Device(torch::kCPU);
    lifuren::Gender model{ nullptr };

public:
    // 加载模型
    // void load(int num_classes, const std::string& modelPath);
    // // 训练模型
    void trian(
        int epoch,
        int batch_size,
        torch::optim::Optimizer& optimizer,
        lifuren::datasets::ImageDatasetType& dataset
    );
    // // 验证模型
    void val(
        int epoch,
        int batch_size,
        lifuren::datasets::ImageDatasetType& dataset
    );
    // // 测试模型
    void test(
        const std::string& data_dir,
        const std::string& image_type
    );
    // 训练验证
    virtual void trainAndVal(
        int   num_epochs,
        int   batch_size,
        float learning_rate,
        const std::string& data_dir,
        const std::string& image_type,
        const std::string& save_path
    );
    // 模型预测
    int pred(cv::Mat& image);

};

}
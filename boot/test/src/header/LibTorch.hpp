/**
 * LibTorch
 * 
 * @author acgist
 */
#pragma once

#include <string>
#include <vector>

#include "torch/torch.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#include "Logger.hpp"
#include "utils/Layers.hpp"
#include "config/Config.hpp"
#include "utils/Datasets.hpp"

LFR_LOG_FORMAT(at::Tensor);

namespace lifuren {

/**
 * 张量测试
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
    torch::nn::Sequential        classifier{ nullptr };

public:
    GenderImpl(int num_classes = 2);
    torch::Tensor forward(torch::Tensor x);

};
TORCH_MODULE(Gender);

/**
 * 性别识别
 */
class GenderHandler {

public:
    torch::Device device = torch::Device(torch::kCPU);
    lifuren::Gender model{ nullptr };

public:
    // 加载模型
    void load(const std::string& modelPath);
    // // 训练模型
    void trian(
        size_t epoch,
        size_t batch_size,
        torch::optim::Optimizer& optimizer,
        lifuren::datasets::ImageDatasetType& dataset
    );
    // // 验证模型
    void val(
        size_t epoch,
        size_t batch_size,
        lifuren::datasets::ImageDatasetType& dataset
    );
    // // 测试模型
    void test(
        const std::string& data_dir,
        const std::string& image_type
    );
    // 训练验证
    virtual void trainAndVal(
        size_t num_epochs,
        size_t batch_size,
        float  learning_rate,
        const  std::string& data_dir,
        const  std::string& image_type,
        const  std::string& save_path
    );
    // 模型预测
    int pred(cv::Mat& image);

};

}

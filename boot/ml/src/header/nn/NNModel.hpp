/**
 * 模型
 */
#pragma once

#include "torch/torch.h"

namespace lifuren {

/**
 * 模型
 * 
 * @author acgist
 */
class NNModel : public torch::nn::Module {

public:
    /**
     * 构造函数
     */
    NNModel();
    /**
     * 析构函数
     */
    ~NNModel();
    /**
     * 正向传播
     * 
     * @param x 张量
     * 
     * @returns 张量
     */
    torch::Tensor forward(torch::Tensor x);

private:
    /**
     * 卷积层
     * 必须设置为空
     */
    torch::nn::Conv2d conv1 = nullptr;
    /**
     * 卷积层
     */
    torch::nn::Conv2d conv2 = nullptr;
    /**
     * 避免过拟合
     * 不能设置为空
     */
    torch::nn::Dropout2d conv2Drop;
    /**
     * 全连接层
     * 必须设置为空
     */
    torch::nn::Linear fc1 = nullptr;
    /**
     * 全连接层
     */
    torch::nn::Linear fc2 = nullptr;

};

}
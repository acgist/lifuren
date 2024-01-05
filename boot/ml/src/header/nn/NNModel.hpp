/**
 * 模型
 */
#pragma once

#include "torch/torch.h"

namespace lifuren {

class NNModel : public torch::nn::Module {

public:
    NNModel();
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
    torch::nn::Conv2d conv1 = nullptr;
    torch::nn::Conv2d conv2 = nullptr;
    torch::nn::Dropout2d conv2Drop;
    torch::nn::Linear fc1 = nullptr;
    torch::nn::Linear fc2 = nullptr;

};

}
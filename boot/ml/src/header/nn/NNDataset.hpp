/**
 * 数据集
 */
#pragma once

#include <string>

#include "torch/torch.h"

namespace lifuren {

/**
 * 数据集
 * 
 * @author acgist
 */
class NNDataset {

public:
    /**
     * @param path 数据路径
     * @param mode 模型类型
     */
    NNDataset(const std::string& path, torch::data::datasets::MNIST::Mode mode);

public:
    /**
     * MNIST数据集
     */
    torch::data::datasets::MNIST mnistDataset;

};

}

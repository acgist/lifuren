/**
 * 数据集
 */
#pragma once

#include <string>

#include "GLog.hpp"

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

/**
 * 训练器
 * 
 * @author acgist
 */
class NNTrainer {

public:
    /**
     * 训练器
     * 
     * @param logInterval 训练打印批次
     */
    NNTrainer(int logInterval);
    /**
     * 训练
     * 
     * @param epoch        训练周期
     * @param model        模型
     * @param optimizer    优化器
     * @param device       训练设备
     * @param trainDataset 数据集
     * @param batchSize    每次训练数据大小
     * @param numWorkers   线程数量
     */
    void train(
        size_t epoch,
        lifuren::NNModel& model,
        torch::optim::Optimizer& optimizer,
        torch::Device device,
        lifuren::NNDataset& trainDataset,
        int batchSize,
        int numWorkers
    );
    /**
     * 测试
     * 
     * @param model       模型
     * @param device      训练设备
     * @param testDataset 数据集
     * @param batchSize   每次测试数据大小
     * @param numWorkers  线程数量
     */
    void test(
        lifuren::NNModel& model,
        torch::Device device,
        lifuren::NNDataset& testDataset,
        int batchSize,
        int numWorkers
    );

private:
    /**
     * 训练打印次数
     */
    int logInterval;

};

}

/**
 * 训练器
 */
#pragma once

#include "torch/torch.h"

#include "GLog.hpp"
#include "NNModel.hpp"
#include "NNDataset.hpp"

namespace lifuren {

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
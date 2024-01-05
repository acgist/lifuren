#pragma once

#include "torch/torch.h"

#include "GLog.hpp"
#include "NNModel.hpp"
#include "NNDataset.hpp"

namespace lifuren {

class NNTrainer {

public:
    NNTrainer(int logInterval);

    /**
     * 训练
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
     */
    void test(
        lifuren::NNModel& model,
        torch::Device device,
        lifuren::NNDataset& testDataset,
        int batchSize,
        int numWorkers
    );

private:
    int logInterval;

};

}
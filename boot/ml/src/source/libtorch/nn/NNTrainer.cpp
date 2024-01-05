#include "../../../header/nn/NNTrainer.hpp"

#include <string>

lifuren::NNTrainer::NNTrainer(int logInterval) : logInterval(logInterval) {
}

void lifuren::NNTrainer::train(
    size_t epoch,
    lifuren::NNModel& model,
    torch::optim::Optimizer& optimizer,
    torch::Device device,
    lifuren::NNDataset& trainDataset,
    int batchSize,
    int numWorkers
) {
    model.train();
    auto dataset = trainDataset.mnistDataset
      // 正则化：均值、标准差
      .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
      .map(torch::data::transforms::Stack<>());
    auto dataLoader = torch::data::make_data_loader(
        dataset,
        torch::data::DataLoaderOptions()
            .batch_size(batchSize)
            .workers(numWorkers)
    );
    auto datasetSize = dataset.size().value();
    size_t batchIndex = 0;
    // 网络训练
    for (const auto& batch : *dataLoader) {
        auto data = batch.data.to(device);
        auto targets = batch.target.to(device);
        optimizer.zero_grad();
        auto output = model.forward(data);
        auto loss = torch::nll_loss(output, targets);
        loss.backward();
        optimizer.step();
        if (batchIndex++ % this->logInterval == 0) {
            LOG(INFO) <<
            "train epoch" << epoch <<
            " " << (batchIndex * batch.data.size(0)) <<
            " / " << datasetSize <<
            " LOSS " << loss.template item<float>() << "\n";
        }
    }
}

void lifuren::NNTrainer::test(
    lifuren::NNModel& model,
    torch::Device device,
    lifuren::NNDataset& testDataset,
    int batchSize,
    int numWorkers
) {
    // 测试设置eval模式
    model.eval();
    double testLoss = 0;
    int32_t correct = 0;
    auto dataset = testDataset.mnistDataset
        .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
        .map(torch::data::transforms::Stack<>());
    auto dataLoader = torch::data::make_data_loader(
        dataset,
        torch::data::DataLoaderOptions()
            .batch_size(batchSize)
            .workers(numWorkers)
    );
    auto datasetSize = dataset.size().value();
    for (const auto& batch : *dataLoader) {
        auto data = batch.data.to(device);
        auto targets = batch.target.to(device);
        auto output = model.forward(data);
        testLoss += torch::nll_loss(
            output,
            targets,
            {},
            torch::Reduction::Sum
        ).item<float>();
        auto pred = output.argmax(1);
        correct += pred.eq(targets).sum().template item<int64_t>();
    }
    testLoss /= datasetSize;
    LOG(INFO) <<
    "avg loss " << testLoss <<
    " Accuracy " << (static_cast<double>(correct) / datasetSize) << "\n";
}

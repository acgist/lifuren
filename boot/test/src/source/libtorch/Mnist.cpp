#include "../../header/Mnist.hpp"

lifuren::NNDataset::NNDataset(
    const std::string& path,
    torch::data::datasets::MNIST::Mode mode
) : mnistDataset(torch::data::datasets::MNIST(path, mode)) {
}

lifuren::NNModel::NNModel() {
    this->conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(1,  10, 5));
    this->conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(10, 20, 5));
    this->fc1 = torch::nn::Linear(320, 50);
    this->fc2 = torch::nn::Linear(50,  10);
    this->register_module("conv1", this->conv1);
    this->register_module("conv2", this->conv2);
    this->register_module("conv2_drop", this->conv2Drop);
    this->register_module("fc1", this->fc1);
    this->register_module("fc2", this->fc2);
}

lifuren::NNModel::~NNModel() {
}

torch::Tensor lifuren::NNModel::forward(torch::Tensor x) {
    x = this->conv1->forward(x);
    x = torch::max_pool2d(x, 2);
    x = torch::relu(x);
    x = this->conv2->forward(x);
    x = this->conv2Drop->forward(x);
    x = torch::max_pool2d(x, 2);
    x = torch::relu(x);
    x = x.view({-1, 320});
    x = this->fc1->forward(x);
    x = torch::relu(x);
    x = torch::dropout(x, 0.5, is_training());
    x = this->fc2->forward(x);
    x = torch::log_softmax(x, 1);
    return x;
}

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
        auto target = batch.target.to(device);
        optimizer.zero_grad();
        auto output = model.forward(data);
        auto loss = torch::nll_loss(output, target);
        loss.backward();
        optimizer.step();
        if (++batchIndex % this->logInterval == 0) {
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
        auto target = batch.target.to(device);
        auto output = model.forward(data);
        testLoss += torch::nll_loss(
            output,
            target,
            {},
            torch::Reduction::Sum
        ).item<float>();
        auto pred = output.argmax(1);
        correct += pred.eq(target).sum().template item<int64_t>();
    }
    testLoss /= datasetSize;
    LOG(INFO) <<
    "avg loss " << testLoss <<
    " Accuracy " << (static_cast<double>(correct) / datasetSize) << "\n";
}

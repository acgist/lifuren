/**
 * 参考文章：https://zhuanlan.zhihu.com/p/638954796
 */
#include "../../header/CNN.hpp"

namespace lifuren {

/**
 * Mnist数据集
 * 
 * @author acgist
 */
class MnistDataset {

public:
    /**
     * @param path 数据路径
     * @param mode 模型类型
     */
    MnistDataset(const std::string& path, torch::data::datasets::MNIST::Mode mode);

public:
    /**
     * MNIST数据集
     */
    torch::data::datasets::MNIST mnistDataset;

};

/**
 * Mnist模型
 * 
 * @author acgist
 */
class MnistModel : public torch::nn::Module {

public:
    MnistModel();
    ~MnistModel();
    /**
     * 正向传播
     * 
     * @param x 张量
     * 
     * @return 张量
     */
    torch::Tensor forward(torch::Tensor x);

private:
    // 卷积层
    torch::nn::Conv2d conv1 = nullptr;
    // 卷积层
    torch::nn::Conv2d conv2 = nullptr;
    // 避免过拟合
    torch::nn::Dropout2d dropout;
    // 全连接层
    torch::nn::Linear fc1 = nullptr;
    // 全连接层
    torch::nn::Linear fc2 = nullptr;

};

/**
 * Mnist训练器
 * 
 * @author acgist
 */
class MnistTrainer {

public:
    /**
     * @param logInterval 训练打印批次
     */
    MnistTrainer(int logInterval);
    /**
     * 训练
     * 
     * @param epoch        训练周期
     * @param workers      线程数量
     * @param batchSize    每次训练数据大小
     * @param model        模型
     * @param trainDataset 数据集
     * @param device       训练设备
     * @param optimizer    优化器
     */
    void train(
        size_t                   epoch,
        int                      workers,
        int                      batchSize,
        torch::Device&           device,
        torch::optim::Optimizer& optimizer,
        lifuren::MnistModel&     model,
        lifuren::MnistDataset&   trainDataset
    );
    /**
     * 测试
     * 
     * @param workers     线程数量
     * @param batchSize   每次测试数据大小
     * @param device      训练设备
     * @param model       模型
     * @param testDataset 数据集
     */
    void test(
        int                    workers,
        int                    batchSize,
        torch::Device&         device,
        lifuren::MnistModel&   model,
        lifuren::MnistDataset& testDataset
    );

private:
    /**
     * 训练打印次数
     */
    int logInterval;

};

}

lifuren::MnistDataset::MnistDataset(
    const std::string& path,
    torch::data::datasets::MNIST::Mode mode
) : mnistDataset(torch::data::datasets::MNIST(path, mode)) {
}

lifuren::MnistModel::MnistModel() {
    this->conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(1,  10, 5));
    this->conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(10, 20, 5));
    this->fc1 = torch::nn::Linear(320, 50);
    this->fc2 = torch::nn::Linear(50,  10);
    this->register_module("conv1", this->conv1);
    this->register_module("conv2", this->conv2);
    this->register_module("dropout", this->dropout);
    this->register_module("fc1", this->fc1);
    this->register_module("fc2", this->fc2);
}

lifuren::MnistModel::~MnistModel() {
}

torch::Tensor lifuren::MnistModel::forward(torch::Tensor x) {
    x = this->conv1->forward(x);
    x = torch::max_pool2d(x, 2);
    x = torch::relu(x);
    x = this->conv2->forward(x);
    x = this->dropout->forward(x);
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

lifuren::MnistTrainer::MnistTrainer(int logInterval) : logInterval(logInterval) {
}

void lifuren::MnistTrainer::train(
    size_t                   epoch,
    int                      workers,
    int                      batchSize,
    torch::Device&           device,
    torch::optim::Optimizer& optimizer,
    lifuren::MnistModel&     model,
    lifuren::MnistDataset&   trainDataset
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
            .workers(workers)
    );
    auto datasetSize = dataset.size().value();
    size_t batchIndex = 0;
    // 网络训练
    for (const auto& batch : *dataLoader) {
        auto data   = batch.data.to(device);
        auto target = batch.target.to(device);
        optimizer.zero_grad();
        auto output = model.forward(data);
        auto loss   = torch::nll_loss(output, target);
        loss.backward();
        optimizer.step();
        if (++batchIndex % this->logInterval == 0) {
            SPDLOG_DEBUG("train epoch {} {} / {} LOSS {}", epoch, batchIndex * batch.data.size(0), datasetSize, loss.template item<float>());
        }
    }
}

void lifuren::MnistTrainer::test(
    int                    workers,
    int                    batchSize,
    torch::Device&         device,
    lifuren::MnistModel&   model,
    lifuren::MnistDataset& testDataset
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
            .workers(workers)
    );
    auto datasetSize = dataset.size().value();
    for (const auto& batch : *dataLoader) {
        auto data   = batch.data.to(device);
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
    SPDLOG_DEBUG("avg loss {} Accuracy {}", testLoss, static_cast<double>(correct) / datasetSize);
}

void lifuren::testMnist() {
    std::string data_root = "D:/tmp/MNIST";
    int workers = 32;
    int logInterval = 10;
    int totalEpochNum = 32;
    int trainBatchSize = 128;
    int testBatchSize = 1024;
    torch::manual_seed(1);
    torch::DeviceType device_type = torch::kCPU;
    if (torch::cuda::is_available()) {
        device_type = torch::kCUDA;
    }
    torch::Device device(device_type);
    lifuren::MnistModel model;
    model.to(device);
    torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));
    lifuren::MnistDataset trainDataset(data_root, torch::data::datasets::MNIST::Mode::kTrain);
    lifuren::MnistDataset testDataset(data_root, torch::data::datasets::MNIST::Mode::kTest);
    auto trainer = lifuren::MnistTrainer(logInterval);
    for (int epoch = 1; epoch <= totalEpochNum; ++epoch) {
        trainer.train(epoch, workers, trainBatchSize, device, optimizer, model, trainDataset);
        trainer.test(workers, testBatchSize, device, model, testDataset);
    }
    torch::serialize::OutputArchive output;
    model.save(output);
    output.save_to("D:/tmp/model/nn.pt");
}

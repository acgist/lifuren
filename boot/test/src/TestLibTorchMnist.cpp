#include "../src/header/Mnist.hpp"

int main(const int argc, const char * const argv[]) {
    lifuren::init(argc, argv);
    std::string data_root = "D:/tmp/MNIST";
    int numWorkers = 32;
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
    lifuren::NNModel model;
    model.to(device);
    torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));
    lifuren::NNDataset trainDataset(data_root, torch::data::datasets::MNIST::Mode::kTrain);
    lifuren::NNDataset testDataset(data_root, torch::data::datasets::MNIST::Mode::kTest);
    auto trainer = lifuren::NNTrainer(logInterval);
    for (int epoch = 1; epoch <= totalEpochNum; ++epoch) {
        trainer.train(epoch, model, optimizer, device, trainDataset, trainBatchSize, numWorkers);
        trainer.test(model, device, testDataset, testBatchSize, numWorkers);
    }
    torch::serialize::OutputArchive output;
    model.save(output);
    output.save_to("D:/tmp/model/nn.pt");
    return 0;
}

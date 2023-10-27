#include "../../header/LibTorch.hpp"

void lifuren::testLinearRegression() {
    const size_t numEpochs    = 60;
    const double learningRate = 0.001;
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    at::Tensor xTrain = torch::randint(0, 10, {15, 1}, torch::TensorOptions(torch::kFloat).device(device));
    at::Tensor yTrain = torch::randint(0, 10, {15, 1}, torch::TensorOptions(torch::kFloat).device(device));
    torch::nn::Linear model(1, 1);
    LOG(INFO) << model->weight;
    // torch::nn::Linear model(10, 5);
    model->to(device);
    // 优化算法：梯度下降
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(learningRate));
    LOG(INFO) << "开始训练";
    for (size_t epoch = 0; epoch < numEpochs; ++epoch) {
        at::Tensor output = model->forward(xTrain);
        // 损失函数
        at::Tensor loss   = torch::nn::functional::mse_loss(output, yTrain);
        optimizer.zero_grad();
        // 计算梯度
        loss.backward();
        optimizer.step();
        if ((epoch + 1) % 5 == 0) {
            LOG(INFO) << "步骤 [" << (epoch + 1) << "/" << numEpochs << "] 损失函数: " << loss.item<double>();
        }
    }
    LOG(INFO) << "训练结束";
}

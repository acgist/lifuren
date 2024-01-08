#include "../../../header/nn/NNModel.hpp"

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

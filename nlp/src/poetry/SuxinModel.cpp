#include "lifuren/poetry/PoetizeModel.hpp"

lifuren::SuxinModuleImpl::SuxinModuleImpl() {
    torch::nn::GRU gru(torch::nn::GRUOptions(1, 1));
    this->gru = register_module("gru", gru);
    torch::nn::Linear linear(torch::nn::LinearOptions(4, 1));
    this->linear = register_module("linear", linear);
}

lifuren::SuxinModuleImpl::~SuxinModuleImpl() {
    unregister_module("gru");
    unregister_module("linear");
}

torch::Tensor lifuren::SuxinModuleImpl::forward(torch::Tensor input) {
    auto [output, hidden] = this->gru->forward(input);
    return this->linear->forward(output.squeeze().t());
}

lifuren::SuxinModel::SuxinModel(lifuren::ModelParams params) : Model(params) {
}

lifuren::SuxinModel::~SuxinModel() {
}

bool lifuren::SuxinModel::defineDataset() {
    return true;
}

void lifuren::SuxinModel::logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) {

}

std::vector<torch::Tensor> lifuren::SuxinModel::pred(std::vector<torch::Tensor> i) {
    return {};
}

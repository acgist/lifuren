#include "lifuren/poetry/PoetizeModel.hpp"

lifuren::SuxinModuleImpl::SuxinModuleImpl() {
    const auto& poetry = lifuren::config::CONFIG.poetry;
    torch::nn::GRU gru(torch::nn::GRUOptions(poetry.size, poetry.size));
    this->gru = register_module("gru", gru);
    torch::nn::Linear linear(torch::nn::LinearOptions(poetry.length + lifuren::config::LIFUREN_POETRY_DATASET_HEAD, 1));
    this->linear = register_module("linear", linear);
}

lifuren::SuxinModuleImpl::~SuxinModuleImpl() {
    unregister_module("gru");
    unregister_module("linear");
}

torch::Tensor lifuren::SuxinModuleImpl::forward(torch::Tensor input) {
    auto [output, hidden] = this->gru->forward(input);
    auto result = this->linear->forward(output.permute({ 2, 1, 0 })).squeeze().t();
    // return torch::log_softmax(result, 1);
    return result;
}

lifuren::SuxinModel::SuxinModel(lifuren::ModelParams params) : Model(params) {
}

lifuren::SuxinModel::~SuxinModel() {
}

bool lifuren::SuxinModel::defineDataset() {
    if(!this->params.train_path.empty()) {
        this->trainDataset = lifuren::dataset::loadPoetryFileDataset(this->params.batch_size, this->params.train_path);
    }
    if(!this->params.val_path.empty()) {
        this->valDataset = lifuren::dataset::loadPoetryFileDataset(this->params.batch_size, this->params.val_path);
    }
    if(!this->params.test_path.empty()) {
        this->testDataset = lifuren::dataset::loadPoetryFileDataset(this->params.batch_size, this->params.test_path);
    }
    return true;
}

void lifuren::SuxinModel::logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) {
    feature = feature.permute({ 1, 0, 2 });
    pred = this->model->forward(feature);
    loss = this->loss(pred, label);
}

#include "lifuren/poetry/Poetry.hpp"

lifuren::poetry::SuxinModuleImpl::SuxinModuleImpl() {
    const auto& poetry = lifuren::config::CONFIG.poetry;
    torch::nn::GRU gru(torch::nn::GRUOptions(poetry.dims, poetry.dims));
    this->gru = register_module("gru", gru);
    torch::nn::Linear linear(torch::nn::LinearOptions(poetry.length + 3, 1));
    this->linear = register_module("linear", linear);
}

lifuren::poetry::SuxinModuleImpl::~SuxinModuleImpl() {
    unregister_module("gru");
    unregister_module("linear");
}

torch::Tensor lifuren::poetry::SuxinModuleImpl::forward(torch::Tensor input) {
    auto [output, hidden] = this->gru->forward(input);
    auto result = this->linear->forward(output.permute({ 2, 1, 0 })).squeeze().t();
    // return torch::log_softmax(result, 1);
    return result;
}

lifuren::poetry::SuxinModel::SuxinModel(lifuren::config::ModelParams params) : Model(params) {
}

lifuren::poetry::SuxinModel::~SuxinModel() {
}

bool lifuren::poetry::SuxinModel::defineDataset() {
    if(lifuren::file::exists(this->params.train_path)) {
        this->trainDataset = lifuren::poetry::loadFileDatasetLoader(this->params.batch_size, this->params.train_path);
    }
    if(lifuren::file::exists(this->params.val_path)) {
        this->valDataset = lifuren::poetry::loadFileDatasetLoader(this->params.batch_size, this->params.val_path);
    }
    if(lifuren::file::exists(this->params.test_path)) {
        this->testDataset = lifuren::poetry::loadFileDatasetLoader(this->params.batch_size, this->params.test_path);
    }
    return true;
}

void lifuren::poetry::SuxinModel::logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) {
    feature = feature.permute({ 1, 0, 2 });
    pred = this->model->forward(feature);
    loss = this->loss(pred, label);
}

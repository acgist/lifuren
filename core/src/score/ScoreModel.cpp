#include "lifuren/ScoreModel.hpp"

#include "lifuren/File.hpp"

lifuren::score::MozartModuleImpl::MozartModuleImpl() {
    // https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
    // input L N input_size = 句子长度 批量数量 词语维度
    torch::nn::GRUOptions gru_options_1(3, 96);
    torch::nn::GRUOptions gru_options_2(96, 3);
    gru_options_1.batch_first(true);
    gru_options_1.num_layers(4);
    gru_options_2.batch_first(true);
    gru_options_2.num_layers(4);
    this->embedding_1 = register_module("embedding_1", torch::nn::Embedding(600, 3));
    this->gru_1 = register_module("gru_1", torch::nn::GRU(gru_options_1));
    this->gru_2 = register_module("gru_2", torch::nn::GRU(gru_options_2));
    this->linear_1 = register_module("linear_1", torch::nn::Linear(3 * 5, 64));
    this->linear_2 = register_module("linear_2", torch::nn::Linear(64, 5));
}

lifuren::score::MozartModuleImpl::~MozartModuleImpl() {
    this->unregister_module("gru_1");
    this->unregister_module("gru_2");
    this->unregister_module("linear_1");
    this->unregister_module("linear_2");
    this->gru_1 = nullptr;
    this->gru_2 = nullptr;
    this->linear_1 = nullptr;
    this->linear_2 = nullptr;
}

torch::Tensor lifuren::score::MozartModuleImpl::forward(torch::Tensor input) {
    // lifuren::logTensor("input1:{}", input.sizes());
    auto [ o1, h1 ] = this->gru_1->forward(this->embedding_1->forward(input));
    // lifuren::logTensor("input2:{}", o1.sizes());
    auto [ o2, h2 ] = this->gru_2->forward(o1);
    // lifuren::logTensor("input3:{}", o2.sizes());
    auto output = torch::flatten(o2, 1, 2);
    // lifuren::logTensor("input4:{}", output.sizes());
    output = this->linear_1->forward(output);
    // lifuren::logTensor("input5:{}", output.sizes());
    output = this->linear_2->forward(output);
    // lifuren::logTensor("input6:{}", output.sizes());
    output = torch::log_softmax(output, 0);
    // lifuren::logTensor("input7:{}", output.sizes());
    return output;
}

lifuren::score::MozartModel::MozartModel(lifuren::config::ModelParams params) : Model(params) {
}

lifuren::score::MozartModel::~MozartModel() {
}

void lifuren::score::MozartModel::defineDataset() {
    if(lifuren::file::exists(this->params.train_path)) {
        this->trainDataset = lifuren::dataset::score::loadMozartDatasetLoader(this->params.batch_size, this->params.train_path);
    }
    if(lifuren::file::exists(this->params.val_path)) {
        this->valDataset = lifuren::dataset::score::loadMozartDatasetLoader(this->params.batch_size, this->params.val_path);
    }
    if(lifuren::file::exists(this->params.test_path)) {
        this->testDataset = lifuren::dataset::score::loadMozartDatasetLoader(this->params.batch_size, this->params.test_path);
    }
}

void lifuren::score::MozartModel::logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) {
    pred = this->model->forward(feature);
    // lifuren::logTensor("pred loss:{}", pred.sizes());
    // lifuren::logTensor("label loss:{}", label.sizes());
    loss = this->loss->forward(pred, label);
}

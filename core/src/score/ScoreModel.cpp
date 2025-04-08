#include "lifuren/ScoreModel.hpp"

#include "lifuren/File.hpp"

const static int num_layers = 2;
const static int gru_size   = 3;
const static int dims       = 4;

lifuren::score::MozartModuleImpl::MozartModuleImpl() {
    // https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
    // input L N input_size = 句子长度 批量数量 词语维度
    torch::nn::GRUOptions gru_options_1(32, 32);
    torch::nn::GRUOptions gru_options_2(32, 32);
    gru_options_1.batch_first(true);
    gru_options_1.num_layers(num_layers);
    gru_options_2.batch_first(true);
    gru_options_2.num_layers(num_layers);
    this->norm = register_module("norm", torch::nn::BatchNorm1d(dims));
    this->gru_1 = register_module("gru_1", torch::nn::GRU(gru_options_1));
    this->gru_2 = register_module("gru_2", torch::nn::GRU(gru_options_2));
    this->linear_1 = register_module("linear_1", torch::nn::Linear(9, 32));
    // this->linear_2 = register_module("linear_2", torch::nn::Linear(32, 64));
    // this->linear_3 = register_module("linear_3", torch::nn::Linear(64, 32));
    this->linear_4 = register_module("linear_4", torch::nn::Linear(32 * dims, 6));
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
    if(!this->hh1.defined()) {
        auto batch_size = input.sizes()[0];
        this->hh1 = torch::zeros({num_layers, batch_size, 32}).to(input.device());
        this->hh2 = torch::zeros({num_layers, batch_size, 32}).to(input.device());
    }
    // auto [o1, h1] = this->gru_1->forward(input, this->hh1);
    auto output = this->linear_1->forward(this->norm->forward(input));
    // output = this->linear_2->forward(output);
    // output = this->linear_3->forward(output);
    auto [o1, h1] = this->gru_1->forward(output, this->hh1);
    auto [o2, h2] = this->gru_2->forward(o1, this->hh2);
    // lifuren::logTensor("o1", torch::cat({input, o1, o2}, 1).sizes());
    // output = this->linear_1->forward(torch::flatten(torch::cat({input, o1, o2}, 1), 1, 2));
    output = this->linear_4->forward(torch::flatten(o2, 1, 2));
    // return torch::softmax(output, 1);
    // return torch::log_softmax(output, 1);
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
    // lifuren::logTensor("pred  loss", pred.sizes());
    // lifuren::logTensor("label loss", label.sizes());
    loss = this->loss->forward(pred, label);
}

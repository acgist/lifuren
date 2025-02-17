#include "lifuren/audio/AudioModel.hpp"

#include "lifuren/File.hpp"

lifuren::audio::ShikuangModuleImpl::ShikuangModuleImpl() {
    const int dims = LFR_DATASET_PCM_DIM_2 * 2;
    this->linear1 = this->register_module("linear1", torch::nn::Linear(torch::nn::LinearOptions(dims, dims)));
    this->linear2 = this->register_module("linear2", torch::nn::Linear(torch::nn::LinearOptions(dims, dims)));
    this->linear3 = this->register_module("linear3", torch::nn::Linear(torch::nn::LinearOptions(dims, dims)));
    this->gru1 = this->register_module("gru1", torch::nn::GRU(torch::nn::GRUOptions(dims, dims)));
    this->gru2 = this->register_module("gru2", torch::nn::GRU(torch::nn::GRUOptions(dims, dims)));
    this->gru3 = this->register_module("gru3", torch::nn::GRU(torch::nn::GRUOptions(dims, dims)));
}

lifuren::audio::ShikuangModuleImpl::~ShikuangModuleImpl() {
    this->unregister_module("linear1");
    this->unregister_module("linear2");
    this->unregister_module("linear3");
    this->unregister_module("gru1");
    this->unregister_module("gru2");
    this->unregister_module("gru3");
}

torch::Tensor lifuren::audio::ShikuangModuleImpl::forward(torch::Tensor input) {
    // std::cout << input.sizes() << '\n';
    input = input.reshape({LFR_DATASET_PCM_BATCH_SIZE, LFR_DATASET_PCM_DIM_1, LFR_DATASET_PCM_DIM_2 * 2}).permute({1, 0, 2});
    auto output = torch::tanh(input);
    auto [o1, s1] = this->gru1->forward(output);
    auto [o2, s2] = this->gru2->forward(o1);
    auto [o3, s3] = this->gru1->forward(o2);
    return o3.permute({1, 0, 2}).reshape({LFR_DATASET_PCM_BATCH_SIZE, LFR_DATASET_PCM_DIM_1, LFR_DATASET_PCM_DIM_2, 2});
}

lifuren::audio::ShikuangModel::ShikuangModel(lifuren::config::ModelParams params) : Model(params) {
}

lifuren::audio::ShikuangModel::~ShikuangModel() {
}

bool lifuren::audio::ShikuangModel::defineDataset() {
    if(lifuren::file::exists(this->params.train_path)) {
        this->trainDataset = lifuren::audio::loadFileDatasetLoader(
            this->params.batch_size,
            lifuren::file::join({this->params.train_path, lifuren::config::LIFUREN_HIDDEN_FILE, lifuren::config::EMBEDDING_MODEL_FILE}).string()
        );
    }
    if(lifuren::file::exists(this->params.val_path)) {
        this->valDataset = lifuren::audio::loadFileDatasetLoader(
            this->params.batch_size,
            lifuren::file::join({this->params.val_path, lifuren::config::LIFUREN_HIDDEN_FILE, lifuren::config::EMBEDDING_MODEL_FILE}).string()
        );
    }
    if(lifuren::file::exists(this->params.test_path)) {
        this->testDataset = lifuren::audio::loadFileDatasetLoader(
            this->params.batch_size,
            lifuren::file::join({this->params.test_path, lifuren::config::LIFUREN_HIDDEN_FILE, lifuren::config::EMBEDDING_MODEL_FILE}).string()
        );
    }
    return true;
}

void lifuren::audio::ShikuangModel::logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) {
    pred = this->model->forward(feature);
    loss = this->loss->forward(pred, label);
}

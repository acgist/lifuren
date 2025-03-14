#include "lifuren/audio/AudioModel.hpp"

#include "lifuren/File.hpp"

lifuren::audio::BachModuleImpl::BachModuleImpl() {
}

lifuren::audio::BachModuleImpl::~BachModuleImpl() {
}

torch::Tensor lifuren::audio::BachModuleImpl::forward(torch::Tensor input) {
    return {};
}

lifuren::audio::BachModel::BachModel(lifuren::config::ModelParams params) : Model(params) {
}

lifuren::audio::BachModel::~BachModel() {
}

bool lifuren::audio::BachModel::defineDataset() {
    return true;
}

void lifuren::audio::BachModel::logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) {
}

lifuren::audio::ShikuangModuleImpl::ShikuangModuleImpl() {
    this->norm = this->register_module("norm", torch::nn::BatchNorm2d(LFR_AUDIO_PCM_DIM_1));
    this->conv1 = this->register_module("conv1", torch::nn::Conv1d(torch::nn::Conv1dOptions(14, 32, 3)));
    this->conv2 = this->register_module("conv2", torch::nn::Conv1d(torch::nn::Conv1dOptions(32, 64, 3)));
    this->linear1 = this->register_module("linear1", torch::nn::Linear(64, 14));
}

lifuren::audio::ShikuangModuleImpl::~ShikuangModuleImpl() {
    this->unregister_module("norm");
    this->unregister_module("conv1");
    this->unregister_module("conv2");
    this->unregister_module("linear1");
}

torch::Tensor lifuren::audio::ShikuangModuleImpl::forward(torch::Tensor input) {
    // std::cout << input.sizes() << '\n';
    std::cout << "=0" << input.sizes() << '\n';
    input = this->norm->forward(input);
    input = input.reshape({100, LFR_AUDIO_PCM_DIM_1, 14});
    input = input.permute({0, 2, 1});
    input = this->conv1->forward(input);
    input = this->conv2->forward(input);
    input = input.permute({0, 2, 1});
    return input.reshape({100, LFR_AUDIO_PCM_DIM_1, 7, 2});
}

lifuren::audio::ShikuangModel::ShikuangModel(lifuren::config::ModelParams params) : Model(params) {
}

lifuren::audio::ShikuangModel::~ShikuangModel() {
}

bool lifuren::audio::ShikuangModel::defineDataset() {
    if(lifuren::file::exists(this->params.train_path)) {
        this->trainDataset = lifuren::dataset::audio::loadShikuangDatasetLoader(
            this->params.batch_size,
            lifuren::file::join({this->params.train_path, lifuren::config::LIFUREN_HIDDEN_FILE, lifuren::config::LIFUREN_EMBEDDING_FILE}).string()
        );
    }
    if(lifuren::file::exists(this->params.val_path)) {
        this->valDataset = lifuren::dataset::audio::loadShikuangDatasetLoader(
            this->params.batch_size,
            lifuren::file::join({this->params.val_path, lifuren::config::LIFUREN_HIDDEN_FILE, lifuren::config::LIFUREN_EMBEDDING_FILE}).string()
        );
    }
    if(lifuren::file::exists(this->params.test_path)) {
        this->testDataset = lifuren::dataset::audio::loadShikuangDatasetLoader(
            this->params.batch_size,
            lifuren::file::join({this->params.test_path, lifuren::config::LIFUREN_HIDDEN_FILE, lifuren::config::LIFUREN_EMBEDDING_FILE}).string()
        );
    }
    return true;
}

void lifuren::audio::ShikuangModel::logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) {
    pred = this->model->forward(feature);
    loss = this->loss->forward(pred, label);
}

lifuren::audio::BeethovenModuleImpl::BeethovenModuleImpl() {
}

lifuren::audio::BeethovenModuleImpl::~BeethovenModuleImpl() {
}

torch::Tensor lifuren::audio::BeethovenModuleImpl::forward(torch::Tensor input) {
    return {};
}

lifuren::audio::BeethovenModel::BeethovenModel(lifuren::config::ModelParams params) : Model(params) {
}

lifuren::audio::BeethovenModel::~BeethovenModel() {
}

bool lifuren::audio::BeethovenModel::defineDataset() {
    return true;
}

void lifuren::audio::BeethovenModel::logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) {
}

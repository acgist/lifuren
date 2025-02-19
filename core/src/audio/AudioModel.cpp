#include "lifuren/audio/AudioModel.hpp"

#include "lifuren/File.hpp"

lifuren::audio::ShikuangModuleImpl::ShikuangModuleImpl() {
    torch::nn::GRUOptions gc1(LFR_DATASET_PCM_DIM_2, LFR_DATASET_PCM_DIM_2);
    torch::nn::GRUOptions gc2(LFR_DATASET_PCM_DIM_2, LFR_DATASET_PCM_DIM_2);
    torch::nn::GRUOptions gc3(28, 112);
    torch::nn::GRUOptions gc4(112, 14);
    gc1.batch_first() = true;
    gc2.batch_first() = true;
    gc3.batch_first() = true;
    gc4.batch_first() = true;
    this->gru1 = this->register_module("gru1", torch::nn::GRU(gc1));
    this->gru2 = this->register_module("gru2", torch::nn::GRU(gc2));
    this->gru3 = this->register_module("gru3", torch::nn::GRU(gc3));
    this->gru4 = this->register_module("gru4", torch::nn::GRU(gc4));
}

lifuren::audio::ShikuangModuleImpl::~ShikuangModuleImpl() {
    this->unregister_module("gru1");
    this->unregister_module("gru2");
    this->unregister_module("gru3");
    this->unregister_module("gru4");
}

torch::Tensor lifuren::audio::ShikuangModuleImpl::forward(torch::Tensor input) {
    // std::cout << input.sizes() << '\n';
    std::cout << "=0" << input.sizes() << '\n';
    auto list = input.split(1, 3);
    auto input0 = list[0].squeeze();
    auto input1 = list[1].squeeze();
    std::cout << "=1" << input0.sizes() << '\n';
    std::cout << "=2" << input1.sizes() << '\n';
    auto [o1, s1] = this->gru1->forward(input0);
    auto [o2, s2] = this->gru2->forward(input1);
    auto o = torch::stack({o1, o2, input0, input1}, 3).reshape({LFR_DATASET_PCM_BATCH_SIZE, LFR_DATASET_PCM_DIM_1, 28});
    auto [o3, s3] = this->gru3->forward(o);
    auto [o4, s4] = this->gru4->forward(o3);
    return o4.reshape({LFR_DATASET_PCM_BATCH_SIZE, LFR_DATASET_PCM_DIM_1, 7, 2});
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

#include "lifuren/audio/ComposeModel.hpp"

#include "lifuren/File.hpp"
#include "lifuren/Tensor.hpp"

lifuren::ShikuangModuleImpl::ShikuangModuleImpl() {
    // [200, 2, 201, 5]
    this->downsample = register_module("downsample", torch::nn::Conv2d(torch::nn::Conv2dOptions(2, 4, {3, 1}).stride({2, 1}).bias(true)));
    this->relu  = register_module("relu",  torch::nn::ReLU());
    this->dropout  = register_module("dropout",  torch::nn::Dropout());
    this->upsample1  = register_module("upsample1",  torch::nn::Linear(torch::nn::LinearOptions(5, 10)));
    this->upsample2  = register_module("upsample2",  torch::nn::Linear(torch::nn::LinearOptions(5, 10)));
    this->upsample3  = register_module("upsample3",  torch::nn::Linear(torch::nn::LinearOptions(5, 10)));
    this->upsample4  = register_module("upsample4",  torch::nn::Linear(torch::nn::LinearOptions(5, 10)));
    this->upsample5  = register_module("upsample5",  torch::nn::Linear(torch::nn::LinearOptions(10, 5)));
    this->upsample6  = register_module("upsample6",  torch::nn::Linear(torch::nn::LinearOptions(100, 201)));
    // torch::nn::Upsample
}

lifuren::ShikuangModuleImpl::~ShikuangModuleImpl() {
    unregister_module("downsample");
    unregister_module("relu");
    unregister_module("dropout");
    unregister_module("upsample1");
    unregister_module("upsample2");
    unregister_module("upsample3");
    unregister_module("upsample4");
    unregister_module("upsample5");
    unregister_module("upsample6");
}

torch::Tensor lifuren::ShikuangModuleImpl::forward(torch::Tensor input) {
    // lifuren::logTensor("output", input.sizes());
    auto output = this->downsample->forward(input);
    // lifuren::logTensor("output", output.sizes());
    // lifuren::logTensor("output", output.slice(1, 0, 1).sizes());
    // [100, 4, 100, 5]
    output = this->dropout(this->relu(output));
    output = torch::stack({
        this->upsample1->forward(output.slice(1, 0, 1)).add(this->upsample2->forward(output.slice(1, 1, 2))),
        this->upsample3->forward(output.slice(1, 2, 3)).mul(this->upsample4->forward(output.slice(1, 3, 4)))
    }).squeeze();
    output = this->dropout(this->relu(output));
    // lifuren::logTensor("output", output.sizes());
    output = this->upsample5->forward(output).permute({1, 0, 2, 3});
    // lifuren::logTensor("output", output.sizes());
    output = this->upsample6->forward(output.permute({0, 1, 3, 2})).permute({0, 1, 3, 2});
    // lifuren::logTensor("output", output.sizes());
    lifuren::logTensor("output", output.sum());
    return output;
}

lifuren::ShikuangModel::ShikuangModel(lifuren::config::ModelParams params) : Model(params) {
}

lifuren::ShikuangModel::~ShikuangModel() {
}

bool lifuren::ShikuangModel::defineDataset() {
    if(lifuren::file::exists(this->params.train_path)) {
        this->trainDataset = lifuren::dataset::loadAudioFileStyleDataset(this->params.batch_size, this->params.train_path);
    }
    if(lifuren::file::exists(this->params.val_path)) {
        this->valDataset = lifuren::dataset::loadAudioFileStyleDataset(this->params.batch_size, this->params.val_path);
    }
    if(lifuren::file::exists(this->params.test_path)) {
        this->testDataset = lifuren::dataset::loadAudioFileStyleDataset(this->params.batch_size, this->params.test_path);
    }
    return true;
}

void lifuren::ShikuangModel::logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) {
    pred = this->model->forward(feature);
    loss = this->loss(pred, label);
}

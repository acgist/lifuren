#include "lifuren/audio/AudioModel.hpp"

#include "lifuren/File.hpp"

lifuren::audio::ShikuangModuleImpl::ShikuangModuleImpl() {
    // [200, 2, 201, 5]
    this->downsample = this->register_module("downsample", torch::nn::Conv2d(torch::nn::Conv2dOptions(2, 4, {3, 1}).stride({2, 1}).bias(true)));
    // this->norm1      = this->register_module("norm1",      torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(4).eps(1e-06).momentum(0.1)));
    // this->norm2      = this->register_module("norm2",      torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(2).eps(1e-06).momentum(0.1)));
    // this->norm3      = this->register_module("norm3",      torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(2).eps(1e-06).momentum(0.1)));
    this->norm1      = this->register_module("norm1",      torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(1)));
    this->norm2      = this->register_module("norm2",      torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(2)));
    this->norm3      = this->register_module("norm3",      torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(2)));
    this->upsample1  = this->register_module("upsample1",  torch::nn::Linear(torch::nn::LinearOptions(5, 10)));
    this->upsample2  = this->register_module("upsample2",  torch::nn::Linear(torch::nn::LinearOptions(5, 10)));
    this->upsample3  = this->register_module("upsample3",  torch::nn::Linear(torch::nn::LinearOptions(5, 10)));
    this->upsample4  = this->register_module("upsample4",  torch::nn::Linear(torch::nn::LinearOptions(5, 10)));
    this->upsample5  = this->register_module("upsample5",  torch::nn::Linear(torch::nn::LinearOptions(10, 5)));
    this->upsample6  = this->register_module("upsample6",  torch::nn::Linear(torch::nn::LinearOptions(100, 201)));
    for(auto& parameter : this->named_parameters()) {
        // torch::nn::init::normal_(parameter.value(), 10.0, 1.0);
        // torch::nn::init::xavier_normal_(parameter);
        // torch::nn::init::cxavier_uniform_(parameter.value(), 1.0);
    }
}

lifuren::audio::ShikuangModuleImpl::~ShikuangModuleImpl() {
    this->unregister_module("downsample");
    this->unregister_module("norm1");
    this->unregister_module("norm2");
    this->unregister_module("norm3");
    this->unregister_module("upsample1");
    this->unregister_module("upsample2");
    this->unregister_module("upsample3");
    this->unregister_module("upsample4");
    this->unregister_module("upsample5");
    this->unregister_module("upsample6");
}

torch::Tensor lifuren::audio::ShikuangModuleImpl::forward(torch::Tensor input) {
    auto input1 = input.slice(1, 0, 1);
    auto input2 = input.slice(1, 1, 2);
    // lifuren::logTensor("output", input.sizes());
    // lifuren::logTensor("output", input1.sizes());
    // lifuren::logTensor("output", input2.sizes());
    // auto output = this->downsample->forward(input);
    // lifuren::logTensor("output", output);
    // lifuren::logTensor("output", output.sizes());
    input2 = this->norm1->forward(input2);
    input2 = torch::tanh(input2);
    // lifuren::logTensor("output", output.slice(1, 0, 1).sizes());
    // [100, 4, 100, 5]
    // output = torch::stack({
    //     this->upsample1->forward(output.slice(1, 0, 1)).add(this->upsample2->forward(output.slice(1, 1, 2))),
    //     this->upsample3->forward(output.slice(1, 2, 3)).add(this->upsample4->forward(output.slice(1, 3, 4)))
    // }, 1).squeeze();
    auto output1 = this->upsample1->forward(input1);
    auto output2 = this->upsample2->forward(input2);
    auto output  = torch::cat({
        output1,
        output2
    }, 1);
    // lifuren::logTensor("output", output.sizes());
    output = this->norm2->forward(output);
    output = torch::tanh(output);
    // lifuren::logTensor("output", output.sizes());
    output = this->upsample5->forward(output);
    // lifuren::logTensor("output", output.sizes());
    // lifuren::logTensor("output", output);
    // output = this->norm3->forward(output);
    // output = torch::tanh(output);
    // output = this->upsample6->forward(output.permute({0, 1, 3, 2})).permute({0, 1, 3, 2});
    // lifuren::logTensor("output", output.sizes());
    // lifuren::logTensor("output", output.sum());
    return output;
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

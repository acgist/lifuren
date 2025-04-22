#include "lifuren/Test.hpp"

#include <memory>
#include <random>

#include "torch/nn.h"
#include "torch/optim.h"

#include "lifuren/Model.hpp"

class LinearModuleImpl : public torch::nn::Module {

private:
    torch::nn::Linear linear{ nullptr };

public:
    LinearModuleImpl() {
        this->linear = register_module("linear", torch::nn::Linear(torch::nn::LinearOptions(1, 1)));
    }

    torch::Tensor forward(torch::Tensor x) {
        return this->linear->forward(x);
    }

    ~LinearModuleImpl() {
        unregister_module("linear");
    }

};

TORCH_MODULE(LinearModule);

// class LinearModel : public lifuren::Model<torch::nn::MSELoss, torch::optim::Adam, LinearModule, lifuren::dataset::RndDatasetLoader> {
class LinearModel : public lifuren::Model<torch::nn::MSELoss, torch::optim::AdamW, LinearModule, lifuren::dataset::RndDatasetLoader> {

public:
    LinearModel(lifuren::config::ModelParams params = {
        .lr         = 0.1F,
        // .lr      = 0.01F,
        // .lr      = 0.001F,
        .batch_size = 10,
        .epoch_size = 256
    }) : Model(params) {
    }

    ~LinearModel() {
    }

public:
    void defineWeight() override {
        for(auto& parameter : this->model->named_parameters()) {
            torch::nn::init::ones_(parameter.value());
        }
    }

    void defineDataset() override {
        std::random_device device;
        std::mt19937 rand(device());
        std::normal_distribution<float> w(100, 10);
        std::normal_distribution<float> b(0.4, 0.2);
        std::vector<torch::Tensor> labels;
        std::vector<torch::Tensor> features;
        labels.reserve(200);
        features.reserve(200);
        // w * 15.4 + 4 + r
        for(int index = 0; index < 200; ++index) {
            float f = w(rand);
            float l = 15.4 * f + 4 + b(rand);
            labels.push_back(torch::tensor({ l }));
            features.push_back(torch::tensor( { f } ));
        }
        auto dataset = lifuren::dataset::Dataset(labels, features).map(torch::data::transforms::Stack<>());
        this->trainDataset = torch::data::make_data_loader<LFT_RND_SAMPLER>(std::move(dataset), this->params.batch_size);
    }

};

[[maybe_unused]] static void testTrain() {
    LinearModel linear;
    linear.define();
    linear.trainValAndTest(false, false);
    // w * 15.4 + 4 + r
    auto output = linear.pred(torch::tensor({ 3.0F }, torch::kFloat32));
    float pred = output.template item<float>();
    SPDLOG_DEBUG("当前预测：{}", pred);
    linear.print(true);
    linear.save();
}

[[maybe_unused]] static void testPred() {
    LinearModel linear;
    linear.load();
    linear.print(true);
    // w * 15.4 + 4 + r
    auto output = linear.pred(torch::tensor({ 3.0F }, torch::kFloat32));
    float pred = output.template item<float>();
    SPDLOG_DEBUG("当前预测：{}", pred);
}

LFR_TEST(
    testTrain();
    // testPred();
);

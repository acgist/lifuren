#include "lifuren/Test.hpp"

#include <random>
#include <memory>

#include "lifuren/Model.hpp"
#include "lifuren/Layer.hpp"
#include "lifuren/Tensor.hpp"
#include "lifuren/Dataset.hpp"

class LinearModuleImpl : public torch::nn::Module {

private:
    torch::nn::Linear linear{ nullptr };

public:
    LinearModuleImpl() {
        linear = register_module("linear", torch::nn::Linear(torch::nn::LinearOptions(1, 1)));
    }
    torch::Tensor forward(torch::Tensor x) {
        return this->linear->forward(x);
    }
    virtual ~LinearModuleImpl() {
        unregister_module("linear");
    }

};

TORCH_MODULE(LinearModule);

class LinearModel : public lifuren::Model<
    lifuren::dataset::RawDatasetLoader,
    float,
    torch::Tensor,
    torch::nn::MSELoss,
    LinearModule,
    torch::optim::SGD
> {

public:
    LinearModel(lifuren::ModelParams params = {
        .lr = 0.001F,
        .batch_size = 10,
        .epoch_count = 256
    }) : Model(params) {
    }
    virtual ~LinearModel() {
    }

public:
    bool defineDataset() override {
        std::random_device device;
        std::mt19937 rand(device());
        std::normal_distribution<float> w(10, 2);
        std::normal_distribution<float> b(0.5, 0.2);
        std::vector<float> labels;
        std::vector<std::vector<float>> features;
        labels.reserve(200);
        features.reserve(200);
        // w * 15.4 + 4 + r
        for(int index = 0; index < 200; ++index) {
            float f = w(rand);
            labels.push_back(15.4 * f + 4 + b(rand));
            features.push_back(std::vector<float>{ f });
        }
        this->trainDataset = std::move(lifuren::dataset::loadRawDataset(this->params.batch_size, labels, features));
        return true;
    }
    float pred(torch::Tensor i) {
        auto o = this->model->forward(i);
        return o.template item<float>();
    }

};

[[maybe_unused]] static void testLine() {
    LinearModel linear;
    linear.define();
    linear.trainValAndTest(false, false);
    float pred = linear.pred(torch::tensor({ 3.0F }, torch::kFloat32));
    SPDLOG_DEBUG("当前预测：{}", pred);
    linear.print();
    linear.save();
}

[[maybe_unused]] static void testLoad() {
    LinearModel model;
    model.define();
    model.load();
    // model.load(lifuren::config::CONFIG.tmp);
    model.print();
    // w * 15.4 + 4 + r
    float pred = model.pred(torch::tensor({ 3.0F }, torch::kFloat32));
    SPDLOG_DEBUG("当前预测：{}", pred);
}

LFR_TEST(
    testLine();
    // testLoad();
);

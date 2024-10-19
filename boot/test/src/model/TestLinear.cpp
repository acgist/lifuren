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

class LinearModel : public lifuren::Model<lifuren::dataset::RawDatasetLoader, float, torch::Tensor, torch::nn::MSELoss, LinearModule> {

public:
    LinearModel(lifuren::ModelParams params = {}) : Model(torch::nn::MSELoss{}, LinearModule{}, params) {
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
        labels.reserve(100);
        features.reserve(100);
        // w * 15.4 + 4 + r
        for(int index = 0; index < 100; ++index) {
            float f = w(rand);
            features.push_back(std::vector<float>{ f });
            labels.push_back(15.4 * f + 4 + b(rand));
        }
        this->trainDataset = std::move(lifuren::dataset::loadRawDataset(5LL, labels, features));
        return true;
    }
    std::shared_ptr<torch::optim::Optimizer> defineOptimizer() override {
        return std::make_shared<torch::optim::SGD>(this->model->parameters(), this->params.lr);
        // return std::make_shared<torch::optim::Adam>(this->model->parameters(), this->params.lr);
    }
    float eval(torch::Tensor i) {
        auto o = this->model->forward(i);
        return o.template item<float>();
    }

};

[[maybe_unused]] static void testLine() {
    LinearModel linear;
    linear.define();
    linear.trainValAndTest(false, false);
    float pred = linear.eval(torch::tensor({3.0F}, torch::kFloat32));
    SPDLOG_DEBUG("当前预测：{}", pred);
    linear.print();
    linear.save();
}

[[maybe_unused]] static void testLoad() {
    // LinearModel model{
    //     // {
    //     //     .batch_size  = 10,
    //     //     .thread_size = 1
    //     // }
    // };
    // model.load(lifuren::config::CONFIG.tmp);
    // // model.loadEval(lifuren::config::CONFIG.tmp);
    // model.print();
    // float data[] { 3.2 };
    // float target[1];
    // // w * 15.4 + 4 + rand
    // float* pred = model.eval(data, target, 1);
    // SPDLOG_DEBUG("当前预测：{}", *pred);
}

LFR_TEST(
    testLine();
    // testLoad();
);

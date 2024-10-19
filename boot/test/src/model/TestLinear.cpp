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
        linear = register_module("linear", torch::nn::Linear(torch::nn::LinearOptions(10, 2)));
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
        std::normal_distribution<> nd(10, 2);
        std::vector<float> labels;
        std::vector<std::vector<float>> features;
        labels.reserve(100);
        features.reserve(100);
        for(int index = 0; index < 100; ++index) {
            labels.push_back(nd(rand));
            std::vector<float> feature(10);
            std::for_each(feature.begin(), feature.end(), [&](auto& v) {
                v = nd(rand);
            });
            features.push_back(feature);
        }
        this->trainDataset = std::move(lifuren::dataset::loadRawDataset(5, labels, features));
        return true;
    }
    std::shared_ptr<torch::optim::Optimizer> defineOptimizer() override {
        return std::make_shared<torch::optim::Adam>(this->model->parameters(), this->params.lr);
    }
    float eval(torch::Tensor i) {
        auto o = this->model->forward(i);
        return o.template item<float>();
    }

};

[[maybe_unused]] static void testLine() {
    std::random_device device;
    std::mt19937 rand(device());
    std::normal_distribution<> weight(10, 2);
    std::normal_distribution<> bias  (0 , 2);
    float features[210];
    float labels  [210];
    for(int index = 0; index < 210; ++index) {
        features[index] = weight(rand);
        labels  [index] = features[index] * 15.4 + 4 + bias(rand);
    }
    // for(int i = 0; i < 210; ++i) {
    //     auto v = features[i];
    //     SPDLOG_DEBUG("d = {}", v);
    // }
    // for(int i = 0; i < 210; ++i) {
    //     auto v = labels[i];
    //     SPDLOG_DEBUG("l = {}", v);
    // }
    lifuren::dataset::RawDataset* dataset = new lifuren::dataset::RawDataset(
        210,
        10,
        features,
        1,
        labels,
        1
    );
    lifuren::Model::OptimizerParams optParams {
        .n_iter = 20
    };
    lifuren::Model::ModelParams params {
        .batch_size  = 10,
        .epoch_count = 256,
        // .epoch_count = 1024,
        .optimizerParams = optParams
    };
    LinearModel save{params};
    save.trainDataset.reset(dataset);
    save.define();
    // save.print();
    save.trainValAndTest(false, false);
    float data[] { 3.2 };
    float target[1];
    // w * 15.4 + 4 + rand
    float* pred = save.eval(data, target, 1);
    SPDLOG_DEBUG("当前预测：{}", *pred);
    SPDLOG_DEBUG("当前权重：{}", save.linear->info());
    lifuren::tensor::print((*save.linear)["linear.weight"]);
    lifuren::tensor::print((*save.linear)["linear.bias"]);
    // save.save(lifuren::config::CONFIG.tmp);
    // save.saveEval(lifuren::config::CONFIG.tmp);
}

[[maybe_unused]] static void testLoad() {
    LinearModel model{
        // {
        //     .batch_size  = 10,
        //     .thread_size = 1
        // }
    };
    model.load(lifuren::config::CONFIG.tmp);
    // model.loadEval(lifuren::config::CONFIG.tmp);
    model.print();
    float data[] { 3.2 };
    float target[1];
    // w * 15.4 + 4 + rand
    float* pred = model.eval(data, target, 1);
    SPDLOG_DEBUG("当前预测：{}", *pred);
}

LFR_TEST(
    testLine();
    // testLoad();
);

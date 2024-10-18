#include "lifuren/Test.hpp"

#include <random>

#include "lifuren/Model.hpp"
#include "lifuren/Dataset.hpp"

class SimpleModuleImpl : public torch::nn::Module {

private:
    torch::nn::Linear linear{ nullptr };

public:
    SimpleModuleImpl() {
        linear = register_module("linear", torch::nn::Linear(torch::nn::LinearOptions(10, 2)));
    }
    torch::Tensor forward(torch::Tensor x) {
        return this->linear->forward(x);
    }
    virtual ~SimpleModuleImpl() {
        unregister_module("linear");
    }

};

TORCH_MODULE(SimpleModule);

class SimpleModel : public lifuren::Model<lifuren::dataset::RawDatasetLoader, float, torch::Tensor, torch::nn::MSELoss, SimpleModule> {

public:
    SimpleModel(lifuren::ModelParams params = {}) : Model(params) {
    }
    ~SimpleModel() {
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
        // TODO: move?
        this->trainDataset = lifuren::dataset::loadRawDataset(5, labels, features);
        return true;
    }
    bool defineModel() override {
        this->model = std::make_shared<SimpleModuleImpl>();
        return true;
    }
    torch::nn::MSELoss defineLoss() override {
        return torch::nn::MSELoss{};
    }
    std::shared_ptr<torch::optim::Optimizer> defineOptimizer() override {
        return std::make_shared<torch::optim::Adam>(this->model->parameters(), this->params.lr);
    }
    float eval(torch::Tensor i) {
        auto o = this->model->forward(i);
        return o.template item<float>();
    }

};

[[maybe_unused]] static void testSaveLoad() {
    SimpleModel save;
    save.define();
    save.save();
    save.print();
    SimpleModel load;
    load.define();
    load.load();
    load.print();
}

[[maybe_unused]] static void testLine() {
    // std::random_device device;
    // std::mt19937 rand(device());
    // std::normal_distribution<> weight(10, 2);
    // std::normal_distribution<> bias  (0 , 2);
    // float features[210];
    // float labels  [210];
    // for(int index = 0; index < 210; ++index) {
    //     features[index] = weight(rand);
    //     labels  [index] = features[index] * 15.4 + 4 + bias(rand);
    // }
    // for(int i = 0; i < 210; ++i) {
    //     auto v = features[i];
    //     SPDLOG_DEBUG("d = {}", v);
    // }
    // for(int i = 0; i < 210; ++i) {
    //     auto v = labels[i];
    //     SPDLOG_DEBUG("l = {}", v);
    // }
    // lifuren::dataset::RawDataset* dataset = new lifuren::dataset::RawDataset{
    //     210,
    //     10,
    //     features,
    //     1,
    //     labels,
    //     1
    // };
    // lifuren::Model::OptimizerParams optParams {
    //     .n_iter = 20
    // };
    // lifuren::Model::ModelParams params {
    //     .batch_size  = 10,
    //     .epoch_count = 256,
    //     .optimizerParams = optParams
    // };
    // SimpleModel save{ params };
    // save.trainDataset.reset(dataset);
    // save.define();
    // // save.print();
    // save.trainValAndTest();
    // float data[] { 3.2 };
    // float target[1];
    // // w * 15.4 + 4 + rand
    // float* pred = save.eval(data, target, 1);
    // SPDLOG_DEBUG("当前预测：{}", *pred);
    // float* w = ggml_get_data_f32(save.fc1_weight);
    // SPDLOG_DEBUG("当前权重：{}", *w);
    // float* b = ggml_get_data_f32(save.fc1_bias);
    // SPDLOG_DEBUG("当前偏置：{}", *b);
}

LFR_TEST(
    testSaveLoad();
    // testLine();
);

#include "lifuren/Test.hpp"

#include <random>

#include "torch/nn.h"
#include "torch/optim.h"

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

class SimpleModel : public lifuren::Model<
    lifuren::dataset::RawDatasetLoader,
    torch::nn::MSELoss,
    torch::optim::Adam,
    SimpleModule
> {

public:
    SimpleModel(lifuren::config::ModelParams params = {}) : Model(params) {
    }
    virtual ~SimpleModel() {
    }

public:
    bool defineDataset() override {
        std::random_device device;
        std::mt19937 rand(device());
        std::normal_distribution<float> nd(10.0F, 2.0F);
        std::vector<torch::Tensor> labels;
        std::vector<torch::Tensor> features;
        labels.reserve(100);
        features.reserve(100);
        for(int index = 0; index < 100; ++index) {
            labels.push_back(torch::tensor({ nd(rand) }));
            std::vector<float> feature(10);
            std::for_each(feature.begin(), feature.end(), [&](auto& v) {
                v = nd(rand);
            });
            features.push_back(torch::from_blob(feature.data(), { 10 }, torch::kFloat32).clone());
        }
        this->trainDataset = std::move(lifuren::dataset::loadRawDataset(5LL, labels, features));
        return true;
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

LFR_TEST(
    testSaveLoad();
);

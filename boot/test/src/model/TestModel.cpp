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

class SimpleModel : public lifuren::Model<lifuren::dataset::RawDatasetLoader, float, torch::Tensor, torch::nn::MSELoss, SimpleModule, torch::optim::Adam> {

public:
    SimpleModel(lifuren::ModelParams params = {}) : Model(params) {
    }
    virtual ~SimpleModel() {
    }

public:
    bool defineDataset() override {
        std::random_device device;
        std::mt19937 rand(device());
        std::normal_distribution<float> nd(10.0F, 2.0F);
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
        this->trainDataset = std::move(lifuren::dataset::loadRawDataset(5LL, labels, features));
        return true;
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

LFR_TEST(
    testSaveLoad();
);

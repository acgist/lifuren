#include "lifuren/Test.hpp"

#include <memory>
#include <random>

#include "torch/nn.h"
#include "torch/optim.h"

#include "lifuren/Model.hpp"
#include "lifuren/Dataset.hpp"

class ClassifyModuleImpl : public torch::nn::Module {

private:
    torch::nn::Linear linear{ nullptr };

public:
    ClassifyModuleImpl() {
        linear = register_module("linear", torch::nn::Linear(torch::nn::LinearOptions(2, 4)));
    }
    torch::Tensor forward(torch::Tensor x) {
        return this->linear->forward(x);
    }
    virtual ~ClassifyModuleImpl() {
        unregister_module("linear");
    }

};

TORCH_MODULE(ClassifyModule);

class ClassifyModel : public lifuren::Model<torch::nn::CrossEntropyLoss, torch::optim::AdamW, ClassifyModule, lifuren::dataset::RndDatasetLoader> {

public:
    ClassifyModel(lifuren::config::ModelParams params = {
        .lr         = 0.001F,
        .batch_size = 100,
        .epoch_size = 8,
        .class_size = 4,
        .classify   = true
    }) : Model(params) {
    }
    virtual ~ClassifyModel() {
    }

public:
    void defineDataset() override {
        std::mt19937 rand(std::random_device{}());
        std::normal_distribution<float> w(10, 2);
        std::normal_distribution<float> b(0.5, 0.2);
        std::vector<torch::Tensor> labels;
        std::vector<torch::Tensor> features;
        labels  .reserve(4000);
        features.reserve(4000);
        for(int index = 0; index < 4000; ++index) {
            int label = index % 4;
            float l[] = { 0, 0, 0, 0 };
            float f[] = { w(rand) * label + b(rand), w(rand) * label + b(rand) };
            l[label]  = 1.0F;
            labels  .push_back(torch::from_blob(l, { 4 }, torch::kFloat32));
            features.push_back(torch::from_blob(f, { 2 }, torch::kFloat32));
        }
        auto dataset = lifuren::dataset::Dataset(labels, features).map(torch::data::transforms::Stack<>());
        this->trainDataset = torch::data::make_data_loader<LFT_RND_SAMPLER>(std::move(dataset), this->params.batch_size);
    }

};

[[maybe_unused]] static void testTrain() {
    ClassifyModel classify;
    classify.define();
    classify.trainValAndTest(false, false);
    classify.print(true);
    classify.save();
    auto pred = torch::log_softmax(classify.pred(torch::tensor({ 3.0F, 4.0F }, torch::kFloat32)).reshape({1, 2}), 1);
    lifuren::logTensor("预测结果", pred);
    auto class_id = pred.argmax(1);
    int class_val = class_id.item<int>();
    SPDLOG_DEBUG("预测结果：{} - {}", class_id.item().toInt(), pred[class_val].item().toFloat());
}

[[maybe_unused]] static void testPred() {
    ClassifyModel classify;
    classify.define();
    classify.load();
    classify.print();
    auto pred = torch::log_softmax(classify.pred(torch::tensor({ 3.0F, 4.0F }, torch::kFloat32)).reshape({1, 2}), 1);
    lifuren::logTensor("当前预测", pred);
    auto class_id = pred.argmax(1);
    int class_val = class_id.item<int>();
    SPDLOG_DEBUG("预测结果：{} - {}", class_id.item().toInt(), pred[class_val].item().toFloat());
}

LFR_TEST(
    testTrain();
    // testPred();
);

#include "lifuren/Test.hpp"

#include <memory>
#include <random>

#include "lifuren/Model.hpp"
#include "lifuren/Dataset.hpp"

class ClassifyModuleImpl : public torch::nn::Module {

private:
    lifuren::config::ModelParams params;
    torch::nn::BatchNorm1d norm    { nullptr };
    torch::nn::Linear      linear_1{ nullptr };
    torch::nn::Linear      linear_2{ nullptr };

public:
    ClassifyModuleImpl(lifuren::config::ModelParams params = {}) : params(params) {
        this->norm     = this->register_module("norm",    torch::nn::BatchNorm1d(2));
        this->linear_1 = this->register_module("linear_1", torch::nn::Linear(torch::nn::LinearOptions( 2, 16)));
        this->linear_2 = this->register_module("linear_2", torch::nn::Linear(torch::nn::LinearOptions(16,  4)));
    }
    torch::Tensor forward(torch::Tensor input) {
        auto output = this->linear_1(this->norm(input));
             output = this->linear_2(torch::relu(output));
        return output;
    }
    virtual ~ClassifyModuleImpl() {
        this->unregister_module("norm");
        this->unregister_module("linear_1");
        this->unregister_module("linear_2");
    }

};

TORCH_MODULE(ClassifyModule);

class ClassifyModel : public lifuren::Model<torch::optim::Adam, ClassifyModule, lifuren::dataset::RndDatasetLoader> {

private:
    torch::nn::CrossEntropyLoss cross_entropy_loss;

public:
    ClassifyModel(lifuren::config::ModelParams params = {
        .lr         = 0.01F,
        .batch_size = 100,
        .epoch_size = 32,
        .class_size = 4,
        .classify   = true
    }) : Model(params) {
    }
    virtual ~ClassifyModel() {
    }

public:
    void defineDataset() override {
        std::mt19937 rand(std::random_device{}());
        std::normal_distribution<float> w(10.0, 1.0); // 标准差越大越难拟合
        std::normal_distribution<float> b( 0.5, 0.2);
        std::vector<torch::Tensor> labels;
        std::vector<torch::Tensor> features;
        labels  .reserve(4000);
        features.reserve(4000);
        for(int index = 0; index < 4000; ++index) {
            int label = index % 4;
            float l[] = { 0, 0, 0, 0 };
            float f[] = { w(rand) * label + b(rand), w(rand) * label + b(rand) };
            l[label]  = 1.0F;
            labels  .push_back(torch::from_blob(l, { 4 }, torch::kFloat32).clone().to(LFR_DTYPE).to(lifuren::getDevice()));
            features.push_back(torch::from_blob(f, { 2 }, torch::kFloat32).clone().to(LFR_DTYPE).to(lifuren::getDevice()));
        }
        auto dataset = lifuren::dataset::Dataset(false, this->params.batch_size, labels, features).map(torch::data::transforms::Stack<>());
        this->trainDataset = torch::data::make_data_loader<LFT_RND_SAMPLER>(std::move(dataset), this->params.batch_size);
    }
    void defineOptimizer() override {
        torch::optim::AdamOptions optims;
        optims.lr (this->params.lr);
        optims.eps(0.0001);
        this->optimizer = std::make_unique<torch::optim::Adam>(this->model->parameters(), optims);
    }
    torch::Tensor loss(torch::Tensor& label, torch::Tensor& pred) {
        return this->cross_entropy_loss->forward(pred, label);
    }

};

[[maybe_unused]] static void testTrain() {
    ClassifyModel classify;
    classify.define();
    classify.trainValAndTest(false, false);
    classify.print(true);
    classify.save();
    auto pred = torch::softmax(classify.pred(torch::tensor({ 4.0F, 4.0F }, torch::kFloat32).reshape({1, 2}).to(LFR_DTYPE).to(lifuren::getDevice())), 1);
    lifuren::logTensor("预测结果", pred);
    auto class_id  = pred.argmax(1);
    auto class_idx = class_id.item<int>();
    SPDLOG_DEBUG("预测结果：{} - {}", class_id.item().toInt(), pred[0][class_idx].item().toFloat());
}

[[maybe_unused]] static void testPred() {
    ClassifyModel classify;
    classify.define();
    classify.load();
    classify.print();
    std::vector<float> data = {
        0.1F,   0.2F,
        2.0F,   1.0F,
        10.0F, 11.0F,
        20.0F, 22.0F,
        30.0F, 33.0F,
        90.0F, 99.0F,
    };
    auto pred = torch::softmax(classify.pred(torch::from_blob(data.data(), { static_cast<int>(data.size()) / 2, 2 }, torch::kFloat32).to(LFR_DTYPE).to(lifuren::getDevice())), 1);
    lifuren::logTensor("当前预测", pred);
    lifuren::logTensor("预测类别", pred.argmax(1));
    lifuren::logTensor("预测类别", std::get<1>(pred.max(1)));
    lifuren::logTensor("预测概率", std::get<0>(pred.max(1)));
}

LFR_TEST(
    testTrain();
    // testPred();
);

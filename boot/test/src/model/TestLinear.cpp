#include "lifuren/Test.hpp"

#include <random>
#include <memory>

#include "ggml.h"

#include "lifuren/Model.hpp"
#include "lifuren/Layer.hpp"
#include "lifuren/Tensor.hpp"
#include "lifuren/Dataset.hpp"

class LinearModel : public lifuren::Model {

public:
    std::unique_ptr<lifuren::layer::Linear> linear{ nullptr };

public:
    LinearModel(lifuren::Model::ModelParams params = {}) : Model(params) {
    }
    ~LinearModel() {
    }

public:
    Model& defineWeight() override {
        this->linear = lifuren::layer::linear(1, 1, this->ctx_weight, this->ctx_compute, "linear");
        this->linear->defineWeight();
        return *this;
    };
    Model& bindWeight(const std::map<std::string, ggml_tensor*> weights) override {
        this->linear = lifuren::layer::linear(1, 1, this->ctx_weight, this->ctx_compute, "linear");
        this->linear->bindWeight(weights);
        return *this;
    };
    ggml_tensor* buildFeatures() override {
        return ggml_new_tensor_2d(this->ctx_compute, GGML_TYPE_F32, 1, this->params.batch_size);
    };
    ggml_tensor* buildLabels() override {
        return ggml_new_tensor_2d(this->ctx_compute, GGML_TYPE_F32, 1, this->params.batch_size);
    };
    ggml_tensor* buildLoss() override {
        // return ggml_sum(this->ctx_compute, ggml_sub(this->ctx_compute, this->logits, this->labels));
        return ggml_sum(this->ctx_compute, ggml_abs(this->ctx_compute, ggml_sub(this->ctx_compute, this->logits, this->labels)));
    };
    ggml_tensor* buildLogits() override {
        return this->linear->forward(this->features);
    };

};

[[maybe_unused]] static void testSaveLoad() {
    lifuren::Model::ModelParams params {
        .batch_size  = 1,
        .epoch_count = 64,
    };
    LinearModel save{params};
    // save.define().print().save(lifuren::config::CONFIG.tmp);
    save.define().print().saveEval(lifuren::config::CONFIG.tmp);
    LinearModel load{params};
    // load.load(lifuren::config::CONFIG.tmp).print();
    load.loadEval(lifuren::config::CONFIG.tmp).print();
}

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
    // testSaveLoad();
);
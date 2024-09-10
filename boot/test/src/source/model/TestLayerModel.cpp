#include "Test.hpp"
#include "lifuren/Model.hpp"

#include <random>
#include <memory>

#include "ggml.h"

#include "lifuren/Layers.hpp"
#include "lifuren/Datasets.hpp"

class LayerModel : public lifuren::Model {

public:
    std::unique_ptr<lifuren::layers::Linear> linear{ nullptr };

public:
    LayerModel(lifuren::Model::ModelParams params = {}) : Model(params) {
    }
    ~LayerModel() {
    }

public:
    Model& defineWeight() override {
        this->linear = lifuren::layers::linear(1, 1, this->ctx_weight, this->ctx_compute, "linear");
        this->linear->defineWeight(this->weights);
        return *this;
    };
    Model& bindWeight() override {
        this->linear = lifuren::layers::linear(1, 1, this->ctx_weight, this->ctx_compute, "linear");
        this->linear->bindWeight(this->weights);
        return *this;
    };
    ggml_tensor* buildDatas() override {
        return ggml_new_tensor_2d(this->ctx_compute, GGML_TYPE_F32, 1, this->params.batch_size);
    };
    ggml_tensor* buildLabels() override {
        return ggml_new_tensor_2d(this->ctx_compute, GGML_TYPE_F32, 1, this->params.batch_size);
    };
    ggml_tensor* buildLoss() override {
        // return ggml_sum(this->ctx_compute, ggml_sub(this->ctx_compute, this->logits, this->labels));
        return ggml_abs(this->ctx_compute, ggml_sub(this->ctx_compute, this->logits, this->labels));
    };
    ggml_tensor* buildLogits() override {
        return this->linear->forward(this->datas);
    };

};

static void testSaveLoad() {
    lifuren::Model::ModelParams params {
        .batch_size  = 1,
        .epoch_count = 64,
    };
    LayerModel save{params};
    save.define().print().save("D:/tmp");
    // save.define().print().saveEval("D:/tmp");
    LayerModel load{params};
    load.load("D:/tmp").print();
    // load.loadEval("D:/tmp").print();
}

static void testLine() {
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
    for(int i = 0; i < 210; ++i) {
        auto v = features[i];
        SPDLOG_DEBUG("d = {}", v);
    }
    for(int i = 0; i < 210; ++i) {
        auto v = labels[i];
        SPDLOG_DEBUG("l = {}", v);
    }
    lifuren::datasets::TensorDataset* dataset = new lifuren::datasets::TensorDataset(
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
        .epoch_count = 64,
        .optimizerParams = optParams
    };
    LayerModel save{params};
    save.define();
    // save.print();
    save.trainDataset.reset(dataset);
    save.trainAndVal();
    float data[] { 3.2 };
    float target[1];
    // w * 15.4 + 4 + rand
    float* pred = save.eval(data, target, 1);
    SPDLOG_DEBUG("当前预测：{}", *pred);
    SPDLOG_DEBUG("当前权重：{}", save.linear->info());
}

LFR_TEST(
    testLine();
    // testSaveLoad();
);

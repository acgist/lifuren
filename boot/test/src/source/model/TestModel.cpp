#include "Test.hpp"
#include "lifuren/Model.hpp"

#include <random>

#include "ggml.h"

#include "lifuren/Datasets.hpp"

class SimpleModel : public lifuren::Model {

public:
    ggml_tensor* fc1_weight{ nullptr };
    ggml_tensor* fc1_bias  { nullptr };

public:
    SimpleModel(lifuren::Model::ModelParams params = {}) : Model(params) {
    }
    ~SimpleModel() {
    }

public:
    Model& defineWeight() override {
        this->fc1_weight = ggml_new_tensor_2d(this->ctx_weight, GGML_TYPE_F32, 1, 1);
        this->fc1_bias   = ggml_new_tensor_1d(this->ctx_weight, GGML_TYPE_F32, 1);
        this->weights.emplace("fc1.weight", this->fc1_weight);
        this->weights.emplace("fc1.bias",   this->fc1_bias);
        return *this;
    };
    Model& bindWeight() override {
        this->fc1_weight = this->weights["fc1.weight"];
        this->fc1_bias   = this->weights["fc1.bias"];
        return *this;
    };
    ggml_tensor* buildDatas() override {
        return ggml_new_tensor_2d(this->ctx_compute, GGML_TYPE_F32, 1, this->params.batch_size);
    }
    ggml_tensor* buildLabels() override {
        return ggml_new_tensor_2d(this->ctx_compute, GGML_TYPE_F32, 1, this->params.batch_size);
    }
    ggml_tensor* buildLoss() override {
        // return ggml_cross_entropy_loss(this->ctx_compute, this->logits, this->labels);
        return ggml_abs(this->ctx_compute, ggml_sub(this->ctx_compute, this->logits, this->labels));
        // return ggml_sum(this->ctx_compute, ggml_sub(this->ctx_compute, this->logits, this->labels));
    };
    ggml_tensor* buildLogits() override {
        return ggml_add(this->ctx_compute,
                ggml_mul_mat(this->ctx_compute, this->fc1_weight, this->datas),
                this->fc1_bias
            );
    };

};

static void testSaveLoad() {
    lifuren::Model::ModelParams params {
        .batch_size  = 1,
        .epoch_count = 64,
    };
    SimpleModel save{params};
    save.define().print().save("D:/tmp");
    // save.define().print().saveEval("D:/tmp");
    SimpleModel load{params};
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
    lifuren::datasets::TensorDataset* dataset = new lifuren::datasets::TensorDataset{
        210,
        10,
        features,
        1,
        labels,
        1
    };
    lifuren::Model::OptimizerParams optParams {
        .n_iter = 20
    };
    lifuren::Model::ModelParams params {
        .batch_size  = 10,
        .epoch_count = 64,
        .optimizerParams = optParams
    };
    SimpleModel save{ params };
    save.trainDataset.reset(dataset);
    save.define();
    // save.print();
    save.trainAndVal();
    float data[] { 3.2 };
    float target[1];
    // w * 15.4 + 4 + rand
    float* pred = save.eval(data, target, 1);
    SPDLOG_DEBUG("当前预测：{}", *pred);
    float* w = ggml_get_data_f32(save.fc1_weight);
    SPDLOG_DEBUG("当前权重：{}", *w);
    float* b = ggml_get_data_f32(save.fc1_bias);
    SPDLOG_DEBUG("当前偏置：{}", *b);
}

LFR_TEST(
    testLine();
    // testSaveLoad();
);
